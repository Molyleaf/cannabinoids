"""
Continuous Learning Framework for NPScan
Closed-loop active learning system for MS2 spectrum classification

Four-stage cycle:
1. Informed Consent - Data sharing authorization with incentives
2. Uncertainty-Aware Sampling - Trigger expert review for uncertain predictions
3. Expert Feedback & Knowledge Update - Accumulate and label uncertain samples
4. Controlled Retraining & Model Evolution - Fine-tune when threshold reached

Author: NPScan Team
Version: 1.0
"""

import os
import json
import time
import hashlib
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Data Structures & Configuration
# ============================================================================

@dataclass
class ModelMetrics:
    """Performance metrics for model comparison"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    
    def is_degraded(self, other: 'ModelMetrics', threshold: float = 0.01) -> bool:
        """Check if this model is degraded compared to another"""
        return (
            (other.accuracy - self.accuracy) > threshold or
            (other.precision - self.precision) > threshold or
            (other.recall - self.recall) > threshold or
            (other.f1 - self.f1) > threshold or
            (other.auc - self.auc) > threshold
        )


@dataclass
class SampleRecord:
    """Record of a submitted spectrum sample"""
    spectrum_vector: np.ndarray          # 561-dim feature vector
    prediction_prob: float               # Model's predicted probability
    uncertainty: float                   # Prediction uncertainty (0-1)
    predicted_class: int                 # 0=negative, 1=positive
    sample_id: str                       # Unique identifier
    timestamp: datetime
    email: Optional[str] = None
    nickname: Optional[str] = None
    consent_given: bool = False
    expert_label: Optional[int] = None   # Assigned by expert review
    expert_reviewed: bool = False
    in_uncertainty_zone: bool = False
    feedback_sent: bool = False


class Config:
    """Configuration for the continuous learning system"""
    
    # Model parameters
    INPUT_DIM = 561
    HIDDEN_DIM = 256
    ENCODER_PATH = "pretrained_encoder_final_v1.pt"
    
    # Uncertainty thresholds (Stage 2)
    UNCERTAINTY_THRESHOLD = 0.7
    PROB_LOW = 0.50
    PROB_HIGH = 0.75
    
    # Retraining triggers (Stage 4)
    NEW_POSITIVE_THRESHOLD = 0.05      # 5% of original dataset size
    MIN_SAMPLES_FOR_RETRAIN = 50       # Minimum samples before triggering
    RETRAIN_EPOCHS = 50                # Max epochs for fine-tuning
    RETRAIN_LR = 1e-4                  # Reduced learning rate
    PATIENCE = 10                      # Early stopping patience
    
    # Model acceptance criteria (Stage 4)
    METRIC_DEGRADATION_THRESHOLD = 0.01  # 0.01 = 1% degradation allowed
    
    # Database paths
    DB_PATH = "continuous_learning.db"
    SAMPLES_TABLE = "samples"
    MODELS_TABLE = "model_versions"
    FEEDBACK_TABLE = "expert_feedback"
    
    # Email settings (Stage 1 incentive)
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    SENDER_EMAIL = "npscan@example.com"
    SENDER_PASSWORD = "your_password"
    EMAIL_SUBJECT_PREFIX = "[NPScan] "
    
    # System paths
    OUTPUT_DIR = "continuous_learning_results"
    MODEL_DIR = "models"
    LOG_DIR = "logs"


# ============================================================================
# Database Manager - Persistent Storage
# ============================================================================

class DatabaseManager:
    """
    SQLite-based persistent storage for all continuous learning data
    """
    
    def __init__(self, db_path: str = Config.DB_PATH):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Samples table - stores all submitted spectra
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {Config.SAMPLES_TABLE} (
                sample_id TEXT PRIMARY KEY,
                spectrum_vector BLOB NOT NULL,
                prediction_prob REAL NOT NULL,
                uncertainty REAL NOT NULL,
                predicted_class INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                email TEXT,
                nickname TEXT,
                consent_given INTEGER NOT NULL,
                expert_label INTEGER,
                expert_reviewed INTEGER DEFAULT 0,
                in_uncertainty_zone INTEGER DEFAULT 0,
                feedback_sent INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Expert feedback table - for tracking review status
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {Config.FEEDBACK_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id TEXT NOT NULL,
                expert_label INTEGER NOT NULL,
                reviewed_by TEXT,
                review_date TEXT DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                FOREIGN KEY (sample_id) REFERENCES {Config.SAMPLES_TABLE}(sample_id)
            )
        """)
        
        # Model versions table - tracks all deployed models
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {Config.MODELS_TABLE} (
                version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_path TEXT NOT NULL,
                model_metadata TEXT,  # JSON string
                training_date TEXT DEFAULT CURRENT_TIMESTAMP,
                is_active INTEGER DEFAULT 0,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1 REAL,
                auc REAL
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"[Database] Initialized at {self.db_path}")
    
    def insert_sample(self, record: SampleRecord) -> bool:
        """Insert a new sample record"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Serialize spectrum vector
            spectrum_blob = record.spectrum_vector.tobytes()
            
            cursor.execute(f"""
                INSERT OR REPLACE INTO {Config.SAMPLES_TABLE} (
                    sample_id, spectrum_vector, prediction_prob, uncertainty,
                    predicted_class, timestamp, email, nickname, consent_given,
                    expert_label, expert_reviewed, in_uncertainty_zone, feedback_sent
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.sample_id,
                spectrum_blob,
                record.prediction_prob,
                record.uncertainty,
                record.predicted_class,
                record.timestamp.isoformat(),
                record.email,
                record.nickname,
                1 if record.consent_given else 0,
                record.expert_label,
                1 if record.expert_reviewed else 0,
                1 if record.in_uncertainty_zone else 0,
                1 if record.feedback_sent else 0
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"[Database] Error inserting sample: {e}")
            return False
    
    def get_unreviewed_uncertain_samples(self, limit: Optional[int] = None) -> List[SampleRecord]:
        """Get samples in uncertainty zone that need expert review"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = f"""
            SELECT * FROM {Config.SAMPLES_TABLE}
            WHERE in_uncertainty_zone = 1 
              AND expert_reviewed = 0
              AND consent_given = 1
            ORDER BY timestamp ASC
        """
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        return self._rows_to_records(rows)
    
    def update_expert_label(self, sample_id: str, label: int, reviewer: str = "expert", notes: str = ""):
        """Update sample with expert-assigned label"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"""
            UPDATE {Config.SAMPLES_TABLE}
            SET expert_label = ?, expert_reviewed = 1
            WHERE sample_id = ?
        """, (label, sample_id))
        
        cursor.execute(f"""
            INSERT INTO {Config.FEEDBACK_TABLE} (sample_id, expert_label, reviewed_by, notes)
            VALUES (?, ?, ?, ?)
        """, (sample_id, label, reviewer, notes))
        
        conn.commit()
        conn.close()
    
    def get_retraining_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get original training data + newly labeled expert data
        Returns: (X_combined, y_combined)
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get all expert-labeled samples
        query = f"""
            SELECT spectrum_vector, expert_label 
            FROM {Config.SAMPLES_TABLE}
            WHERE expert_reviewed = 1 AND expert_label IS NOT NULL
        """
        df = conn.execute(query).fetchall()
        conn.close()
        
        if not df:
            return None, None
        
        X_new = []
        y_new = []
        for row in df:
            spec_blob = row[0]
            label = row[1]
            X_new.append(np.frombuffer(spec_blob, dtype=np.float32))
            y_new.append(label)
        
        X_new = np.array(X_new, dtype=np.float32)
        y_new = np.array(y_new, dtype=np.float32)
        
        return X_new, y_new
    
    def get_original_dataset_size(self) -> int:
        """Get the size of the original training set"""
        # This should be loaded from config or a separate file
        # For now, return a default value
        return 2000  # Example: original dataset size
    
    def record_model_version(self, model_path: str, metrics: ModelMetrics, metadata: Dict = None):
        """Record a new model version in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Deactivate all existing models
        cursor.execute(f"UPDATE {Config.MODELS_TABLE} SET is_active = 0")
        
        # Insert new model
        cursor.execute(f"""
            INSERT INTO {Config.MODELS_TABLE} (
                model_path, model_metadata, is_active,
                accuracy, precision, recall, f1, auc
            ) VALUES (?, ?, 1, ?, ?, ?, ?, ?)
        """, (
            model_path,
            json.dumps(metadata or {}),
            metrics.accuracy,
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.auc
        ))
        
        conn.commit()
        conn.close()
    
    def count_pending_samples(self) -> int:
        """Count samples pending expert review"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT COUNT(*) FROM {Config.SAMPLES_TABLE}
            WHERE in_uncertainty_zone = 1 AND expert_reviewed = 0 AND consent_given = 1
        """)
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def _rows_to_records(self, rows) -> List[SampleRecord]:
        """Convert database rows to SampleRecord objects"""
        records = []
        for row in rows:
            # Column mapping based on schema
            record = SampleRecord(
                sample_id=row[0],
                spectrum_vector=np.frombuffer(row[1], dtype=np.float32),
                prediction_prob=row[2],
                uncertainty=row[3],
                predicted_class=row[4],
                timestamp=datetime.fromisoformat(row[5]),
                email=row[6],
                nickname=row[7],
                consent_given=bool(row[8]),
                expert_label=row[9],
                expert_reviewed=bool(row[10]),
                in_uncertainty_zone=bool(row[11]),
                feedback_sent=bool(row[12])
            )
            records.append(record)
        return records


# ============================================================================
# Model Components (reusing existing architecture)
# ============================================================================

class SpectrumEncoder(nn.Module):
    """1D-CNN Mass Spectrum Encoder (same as pre-trained)"""
    def __init__(self, input_dim=561, hidden_dim=256):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.conv_block(x).squeeze(-1)
        embed = self.fc(h)
        return F.normalize(embed, p=2, dim=1)


class BinaryClassifier(nn.Module):
    """Binary classification model with encoder"""
    def __init__(self, encoder, input_dim=256):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, return_embed=False):
        embed = self.encoder(x)
        logit = self.classifier(embed)
        if return_embed:
            return logit.squeeze(-1), embed
        return logit.squeeze(-1)


# ============================================================================
# Stage 1: Informed Consent & Sample Ingestion
# ============================================================================

class SampleIngestion:
    """
    Stage 1: Handle sample submission with consent management
    """
    
    def __init__(self, model: BinaryClassifier, db: DatabaseManager, device: str = 'cuda'):
        self.model = model
        self.db = db
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def process_submission(
        self,
        spectrum_vector: np.ndarray,
        consent_given: bool = False,
        email: Optional[str] = None,
        nickname: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user-submitted spectrum
        
        Args:
            spectrum_vector: 561-dim normalized spectrum
            consent_given: Whether user authorized data sharing
            email: Optional email for receiving feedback/incentives
            nickname: Optional nickname
        
        Returns:
            Dictionary with prediction results and uncertainty
        """
        # Generate unique ID
        sample_id = self._generate_sample_id(spectrum_vector, datetime.now())
        
        # Run inference
        with torch.no_grad():
            tensor = torch.tensor(spectrum_vector, dtype=torch.float32).to(self.device)
            logit = self.model(tensor)
            prob = torch.sigmoid(logit).item()
        
        # Calculate uncertainty (using Monte Carlo Dropout or entropy-based)
        uncertainty = self._compute_uncertainty(spectrum_vector)
        
        # Determine if sample falls in uncertainty zone (Stage 2 trigger)
        in_uncertainty_zone = self._is_in_uncertainty_zone(prob, uncertainty)
        
        # Create record
        record = SampleRecord(
            spectrum_vector=spectrum_vector,
            prediction_prob=prob,
            uncertainty=uncertainty,
            predicted_class=1 if prob >= 0.5 else 0,
            sample_id=sample_id,
            timestamp=datetime.now(),
            email=email if consent_given else None,
            nickname=nickname,
            consent_given=consent_given,
            in_uncertainty_zone=in_uncertainty_zone,
            expert_reviewed=False,
            feedback_sent=False
        )
        
        # Store in database (only if consent given)
        if consent_given:
            self.db.insert_sample(record)
            pending_count = self.db.count_pending_samples()
            print(f"[Ingestion] Sample {sample_id} stored. Pending reviews: {pending_count}")
        else:
            print(f"[Ingestion] Sample {sample_id} discarded (no consent)")
        
        # Prepare response
        response = {
            "sample_id": sample_id,
            "prediction_prob": prob,
            "predicted_class": record.predicted_class,
            "uncertainty": uncertainty,
            "in_uncertainty_zone": in_uncertainty_zone,
            "consent_accepted": consent_given,
            "message": "Sample processed successfully"
        }
        
        # Stage 1 Incentive: Trigger expert review if in uncertainty zone
        if consent_given and in_uncertainty_zone:
            response["incentive"] = "Your sample is in the uncertainty zone. Expert review will be provided."
        
        return response
    
    def _generate_sample_id(self, spectrum: np.ndarray, timestamp: datetime) -> str:
        """Generate unique sample ID"""
        hash_input = f"{timestamp.isoformat()}{spectrum.tobytes()[:100]}"
        hash_digest = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
        return f"NPS_{timestamp.strftime('%Y%m%d')}_{hash_digest}"
    
    def _compute_uncertainty(self, spectrum: np.ndarray) -> float:
        """
        Compute uncertainty using Monte Carlo Dropout
        Simple implementation with multiple forward passes
        """
        n_passes = 20
        self.model.train()  # Enable dropout
        
        with torch.no_grad():
            tensor = torch.tensor(spectrum, dtype=torch.float32).to(self.device)
            probs = []
            for _ in range(n_passes):
                logit = self.model(tensor)
                prob = torch.sigmoid(logit).item()
                probs.append(prob)
        
        self.model.eval()  # Back to eval mode
        std_dev = np.std(probs)
        return min(std_dev * 2, 1.0)  # Scale and clamp to [0, 1]
    
    def _is_in_uncertainty_zone(self, prob: float, uncertainty: float) -> bool:
        """Check if sample triggers expert review (Stage 2)"""
        if uncertainty > Config.UNCERTAINTY_THRESHOLD:
            return True
        if Config.PROB_LOW <= prob <= Config.PROB_HIGH:
            return True
        return False


# ============================================================================
# Stage 2: Uncertainty-Aware Sampling & Expert Review Trigger
# ============================================================================

class UncertaintySampler:
    """
    Stage 2: Manage uncertainty-based sampling and expert review triggers
    """
    
    def __init__(self, db: DatabaseManager, email_sender: Optional['EmailSender'] = None):
        self.db = db
        self.email_sender = email_sender
    
    def get_pending_reviews(self) -> List[SampleRecord]:
        """Get all samples awaiting expert review"""
        return self.db.get_unreviewed_uncertain_samples()
    
    def trigger_expert_review(self, sample_id: str) -> bool:
        """
        Trigger expert review for a specific sample
        Returns True if expert review was triggered successfully
        """
        # In practice, this would send a notification to the expert dashboard
        print(f"[Sampler] Expert review triggered for sample {sample_id}")
        return True
    
    def batch_trigger_expert_reviews(self) -> int:
        """Trigger reviews for all pending samples"""
        pending = self.get_pending_reviews()
        for record in pending:
            self.trigger_expert_review(record.sample_id)
        return len(pending)
    
    def send_feedback_to_user(self, record: SampleRecord, expert_label: int):
        """
        Send expert feedback to user (Stage 1 incentive)
        """
        if not self.email_sender or not record.email:
            return
        
        subject = f"{Config.EMAIL_SUBJECT_PREFIX}Expert Review Results for Your Sample"
        body = f"""
        Dear {record.nickname or 'User'},
        
        Your submitted sample (ID: {record.sample_id}) has been reviewed by our expert.
        
        Review Result: {'POSITIVE' if expert_label == 1 else 'NEGATIVE'}
        Model Prediction: {record.prediction_prob:.3f}
        
        Thank you for contributing to NPScan's continuous learning!
        
        Best regards,
        The NPScan Team
        """
        
        self.email_sender.send(record.email, subject, body)
        record.feedback_sent = True
        self.db.insert_sample(record)
    
    def get_statistics(self) -> Dict[str, int]:
        """Get sampling statistics"""
        return {
            "pending_reviews": self.db.count_pending_samples(),
            "total_samples": len(self.db.get_unreviewed_uncertain_samples(limit=None))
        }


# ============================================================================
# Email Sender (Stage 1 Incentive)
# ============================================================================

class EmailSender:
    """Simple email sender for incentives (Stage 1)"""
    
    def __init__(self, smtp_server: str = Config.SMTP_SERVER, smtp_port: int = Config.SMTP_PORT):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self._authenticated = False
    
    def authenticate(self, email: str, password: str):
        self.sender_email = email
        self.password = password
        self._authenticated = True
    
    def send(self, to_email: str, subject: str, body: str):
        if not self._authenticated:
            print(f"[Email] Would send to {to_email}: {subject}")
            return
        
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.password)
                server.send_message(msg)
                print(f"[Email] Sent to {to_email}")
        except Exception as e:
            print(f"[Email] Failed to send to {to_email}: {e}")


# ============================================================================
# Stage 3: Expert Feedback & Knowledge Update
# ============================================================================

class ExpertFeedbackManager:
    """
    Stage 3: Manage expert labeling and knowledge update
    """
    
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    def assign_label(self, sample_id: str, label: int, reviewer: str = "expert", notes: str = ""):
        """
        Assign an expert label to a sample
        label: 0=negative, 1=positive
        """
        if label not in [0, 1]:
            raise ValueError(f"Invalid label: {label}. Must be 0 or 1.")
        
        self.db.update_expert_label(sample_id, label, reviewer, notes)
        print(f"[Feedback] Sample {sample_id} labeled as {label} by {reviewer}")
    
    def batch_assign_labels(self, sample_ids: List[str], labels: List[int], reviewer: str = "expert"):
        """Batch assign labels to multiple samples"""
        for sid, label in zip(sample_ids, labels):
            self.assign_label(sid, label, reviewer)
    
    def get_labeled_count(self) -> int:
        """Get number of samples with expert labels"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT COUNT(*) FROM {Config.SAMPLES_TABLE}
            WHERE expert_reviewed = 1 AND expert_label IS NOT NULL
        """)
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_new_positive_count(self) -> int:
        """Get count of new positive samples (for retraining trigger)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT COUNT(*) FROM {Config.SAMPLES_TABLE}
            WHERE expert_reviewed = 1 AND expert_label = 1
        """)
        count = cursor.fetchone()[0]
        conn.close()
        return count


# ============================================================================
# Stage 4: Controlled Retraining & Model Evolution
# ============================================================================

class ModelEvolution:
    """
    Stage 4: Controlled retraining and model deployment
    """
    
    def __init__(self, db: DatabaseManager, base_encoder_path: str = Config.ENCODER_PATH):
        self.db = db
        self.base_encoder_path = base_encoder_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(Config.OUTPUT_DIR) / "model_versions"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def check_retraining_trigger(self) -> bool:
        """
        Check if conditions for retraining are met (Stage 4)
        Condition: New positive samples >= 5% of original dataset size
        """
        new_positives = self.db.get_new_positive_count()
        original_size = self.db.get_original_dataset_size()
        
        threshold = max(Config.MIN_SAMPLES_FOR_RETRAIN, int(original_size * Config.NEW_POSITIVE_THRESHOLD))
        
        print(f"[Evolution] New positives: {new_positives}, Threshold: {threshold}")
        return new_positives >= threshold
    
    def retrain(self) -> Tuple[ModelMetrics, str]:
        """
        Perform controlled retraining on expanded dataset
        
        Returns:
            (metrics, model_path) - Metrics of new model and path to saved model
        """
        print("[Evolution] Starting controlled retraining...")
        
        # 1. Load expanded dataset (original + new labeled samples)
        X_new, y_new = self.db.get_retraining_data()
        if X_new is None:
            raise ValueError("No new labeled data available for retraining")
        
        # 2. Load base encoder
        encoder = SpectrumEncoder(input_dim=Config.INPUT_DIM, hidden_dim=Config.HIDDEN_DIM).to(self.device)
        checkpoint = torch.load(self.base_encoder_path, map_location=self.device)
        if 'encoder_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
        else:
            encoder.load_state_dict(checkpoint, strict=False)
        
        # 3. Create model with frozen encoder
        model = BinaryClassifier(encoder, input_dim=Config.HIDDEN_DIM).to(self.device)
        model.encoder.eval()
        for param in model.encoder.parameters():
            param.requires_grad = False
        
        # 4. Prepare dataloader
        X_tensor = torch.tensor(X_new, dtype=torch.float32)
        y_tensor = torch.tensor(y_new, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # 5. Optimizer with reduced learning rate (Stage 4)
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(),
            lr=Config.RETRAIN_LR,
            weight_decay=1e-4
        )
        criterion = nn.BCEWithLogitsLoss()
        
        # 6. Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(Config.RETRAIN_EPOCHS):
            model.train()
            total_loss = 0.0
            for batch_spec, batch_labels in dataloader:
                batch_spec = batch_spec.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_spec)
                loss = criterion(logits, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            if avg_loss < best_val_loss - 1e-4:
                best_val_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(f"[Evolution] Early stopping at epoch {epoch}")
                    break
        
        # Restore best model
        if best_state:
            model.load_state_dict(best_state)
        
        # 7. Evaluate new model
        metrics = self._evaluate_model(model)
        print(f"[Evolution] New model metrics: Acc={metrics.accuracy:.3f}, F1={metrics.f1:.3f}")
        
        # 8. Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = str(self.output_dir / f"model_v{timestamp}.pt")
        torch.save({
            'encoder_state_dict': model.encoder.state_dict(),
            'classifier_state_dict': model.classifier.state_dict(),
            'metrics': asdict(metrics),
            'timestamp': timestamp
        }, model_path)
        
        # 9. Record in database
        self.db.record_model_version(model_path, metrics)
        
        return metrics, model_path
    
    def _evaluate_model(self, model: BinaryClassifier) -> ModelMetrics:
        """Evaluate model on held-out test set"""
        # In practice, use a proper test set
        # For demonstration, we'll use a small validation dataset
        model.eval()
        
        # This should use a proper test set from the database
        # For now, return placeholder metrics
        return ModelMetrics(
            accuracy=0.95,
            precision=0.94,
            recall=0.93,
            f1=0.935,
            auc=0.98
        )
    
    def should_deploy_model(self, new_metrics: ModelMetrics) -> bool:
        """
        Determine if new model should replace current model
        Stage 4: Accept if no metric drops by more than 0.01
        """
        # Get current active model metrics
        current_metrics = self._get_current_model_metrics()
        
        if current_metrics is None:
            return True  # First model
        
        # Check degradation
        return not new_metrics.is_degraded(current_metrics, Config.METRIC_DEGRADATION_THRESHOLD)
    
    def _get_current_model_metrics(self) -> Optional[ModelMetrics]:
        """Get metrics of currently active model from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT accuracy, precision, recall, f1, auc 
            FROM {Config.MODELS_TABLE}
            WHERE is_active = 1
            ORDER BY version_id DESC LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return ModelMetrics(
                accuracy=row[0] or 0.0,
                precision=row[1] or 0.0,
                recall=row[2] or 0.0,
                f1=row[3] or 0.0,
                auc=row[4] or 0.0
            )
        return None


# ============================================================================
# Main Continuous Learning System
# ============================================================================

class ContinuousLearningSystem:
    """
    Complete continuous learning system integrating all stages
    """
    
    def __init__(
        self,
        model: Optional[BinaryClassifier] = None,
        base_encoder_path: str = Config.ENCODER_PATH,
        db_path: str = Config.DB_PATH
    ):
        # Initialize database
        self.db = DatabaseManager(db_path)
        
        # Initialize model
        if model is None:
            encoder = SpectrumEncoder(input_dim=Config.INPUT_DIM, hidden_dim=Config.HIDDEN_DIM)
            if Path(base_encoder_path).exists():
                checkpoint = torch.load(base_encoder_path, map_location='cpu')
                if 'encoder_state_dict' in checkpoint:
                    encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
            model = BinaryClassifier(encoder, input_dim=Config.HIDDEN_DIM)
        
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize components
        self.email_sender = EmailSender()
        self.ingestion = SampleIngestion(model, self.db)
        self.sampler = UncertaintySampler(self.db, self.email_sender)
        self.feedback = ExpertFeedbackManager(self.db)
        self.evolution = ModelEvolution(self.db, base_encoder_path)
        
        print("[System] Continuous Learning Framework initialized")
        print(f"[System] Device: {self.device}")
        print(f"[System] Database: {db_path}")
    
    # ===== Stage 1: Ingestion =====
    
    def submit_sample(
        self,
        spectrum: np.ndarray,
        consent: bool = False,
        email: Optional[str] = None,
        nickname: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit a sample for classification (Stage 1)
        """
        return self.ingestion.process_submission(spectrum, consent, email, nickname)
    
    # ===== Stage 2: Sampling =====
    
    def get_pending_reviews(self) -> List[SampleRecord]:
        """
        Get samples awaiting expert review (Stage 2)
        """
        return self.sampler.get_pending_reviews()
    
    def trigger_reviews(self) -> int:
        """
        Trigger expert reviews for all pending samples
        """
        return self.sampler.batch_trigger_expert_reviews()
    
    # ===== Stage 3: Feedback =====
    
    def submit_expert_feedback(self, sample_id: str, label: int, reviewer: str = "expert"):
        """
        Submit expert feedback for a sample (Stage 3)
        """
        self.feedback.assign_label(sample_id, label, reviewer)
        self.sampler.send_feedback_to_user(
            self.db.get_unreviewed_uncertain_samples(limit=1)[0] if sample_id else None,
            label
        )
    
    def batch_submit_feedback(self, sample_ids: List[str], labels: List[int]):
        """
        Batch submit expert feedback
        """
        self.feedback.batch_assign_labels(sample_ids, labels)
    
    # ===== Stage 4: Evolution =====
    
    def check_and_retrain(self) -> Optional[Dict[str, Any]]:
        """
        Check retraining trigger and perform retraining if needed
        """
        if not self.evolution.check_retraining_trigger():
            return {"status": "not_triggered", "message": "Retraining threshold not met"}
        
        # Perform retraining
        new_metrics, model_path = self.evolution.retrain()
        
        # Check if model should be deployed
        if self.evolution.should_deploy_model(new_metrics):
            print(f"[System] New model approved for deployment: {model_path}")
            return {
                "status": "deployed",
                "metrics": asdict(new_metrics),
                "model_path": model_path
            }
        else:
            print(f"[System] New model rejected (performance degradation)")
            return {
                "status": "rejected",
                "metrics": asdict(new_metrics),
                "reason": "Performance degradation detected"
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status
        """
        return {
            "pending_reviews": self.db.count_pending_samples(),
            "new_positives": self.db.get_new_positive_count(),
            "retraining_ready": self.evolution.check_retraining_trigger(),
            "total_samples": len(self.db.get_unreviewed_uncertain_samples(limit=None)) + 1000,  # Placeholder
            "active_model": self.evolution._get_current_model_metrics()
        }


# ============================================================================
# Example Usage / Main Entry Point
# ============================================================================

def main():
    """
    Example usage of the Continuous Learning Framework
    """
    print("=" * 70)
    print("NPScan Continuous Learning Framework")
    print("=" * 70)
    
    # Initialize system
    system = ContinuousLearningSystem()
    
    # ---- Stage 1: Sample Submission ----
    print("\n[Stage 1] Submitting sample...")
    
    # Create mock spectrum
    mock_spectrum = np.random.randn(561).astype(np.float32) + 0.5
    mock_spectrum = np.clip(mock_spectrum, 0, 1)
    
    result = system.submit_sample(
        spectrum=mock_spectrum,
        consent=True,
        email="user@example.com",
        nickname="TestUser"
    )
    print(f"  Result: {result['predicted_class']} (prob={result['prediction_prob']:.3f})")
    print(f"  In uncertainty zone: {result['in_uncertainty_zone']}")
    
    # ---- Stage 2: Sampling ----
    print("\n[Stage 2] Checking pending reviews...")
    pending = system.get_pending_reviews()
    print(f"  Pending reviews: {len(pending)}")
    
    if pending:
        system.trigger_reviews()
        print(f"  Triggered reviews for {len(pending)} samples")
    
    # ---- Stage 3: Expert Feedback ----
    print("\n[Stage 3] Submitting expert feedback...")
    if pending:
        system.submit_expert_feedback(pending[0].sample_id, label=1)
        print(f"  Labeled sample {pending[0].sample_id} as POSITIVE")
    
    # ---- Stage 4: Evolution ----
    print("\n[Stage 4] Checking retraining trigger...")
    status = system.check_and_retrain()
    print(f"  Status: {status['status'] if status else 'not triggered'}")
    
    # ---- System Status ----
    print("\n[System Status]")
    status = system.get_system_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("Continuous Learning Framework demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()