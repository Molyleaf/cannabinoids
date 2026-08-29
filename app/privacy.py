# -*- coding: utf-8 -*-
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Data queue and privacy storage directory
QUEUE_DIR = Path(__file__).resolve().parent / "data_queue"
QUEUE_DIR.mkdir(parents=True, exist_ok=True)

UNCERTAIN_QUEUE_FILE = QUEUE_DIR / "uncertain_samples.jsonl"
CONTRIBUTIONS_FILE = QUEUE_DIR / "authorized_contributions.jsonl"

LEGACY_UNCERTAIN_JSON = QUEUE_DIR / "uncertain_samples.json"
LEGACY_CONTRIBUTIONS_JSON = QUEUE_DIR / "authorized_contributions.json"


def _migrate_legacy_json(legacy_path: Path, target_jsonl_path: Path) -> None:
    """Migrate legacy whole-array JSON file to atomic append JSONL format."""
    if not legacy_path.exists():
        return
    try:
        with open(legacy_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        if isinstance(records, list) and records:
            existing_lines = set()
            if target_jsonl_path.exists():
                with open(target_jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            existing_lines.add(line.strip())
            with open(target_jsonl_path, "a", encoding="utf-8") as f:
                for r in records:
                    serialized = json.dumps(r, ensure_ascii=False)
                    if serialized not in existing_lines:
                        f.write(serialized + "\n")
        legacy_path.unlink(missing_ok=True)
    except Exception:
        pass


def _migrate_all_legacy() -> None:
    """Perform one-time initialization migration of legacy JSON files."""
    _migrate_legacy_json(LEGACY_UNCERTAIN_JSON, UNCERTAIN_QUEUE_FILE)
    _migrate_legacy_json(LEGACY_CONTRIBUTIONS_JSON, CONTRIBUTIONS_FILE)


# Run one-time migration upon module load
_migrate_all_legacy()


def _append_jsonl_record(filepath: Path, record: Dict[str, Any]) -> None:
    """Atomically append a single JSON Line record."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(line)


def _count_jsonl_lines(filepath: Path) -> int:
    """Efficiently count lines in a JSONL file with O(1) memory overhead."""
    if not filepath.exists():
        return 0
    count = 0
    with open(filepath, "rb") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def record_sample_if_authorized(
    file_name: str,
    peaks: List,
    model_type: str,
    probability: float,
    predicted_result: str,
    user_consent: bool,
    user_name: str = "",
    user_email: str = ""
) -> Dict[str, Any]:
    """
    Phase 1: Informed consent & privacy check. If not authorized, do not persist or upload data.
    Phase 2: If authorized, check if prediction probability falls into the [0.3, 0.7] uncertainty interval.
    If so, save to local review queue and trigger human review flag.
    """
    if not user_consent:
        return {
            "authorized": False,
            "status": "Data sharing not authorized. The system only outputs detection results and does not retain or upload any data.",
            "is_uncertain": False,
            "review_flag": False
        }

    # Authorized state: record contribution
    contribution_record = {
        "timestamp": datetime.now().isoformat(),
        "file_name": file_name,
        "user_name": user_name.strip() if user_name else "Anonymous Contributor",
        "user_email": user_email.strip() if user_email else "",
        "model_type": model_type,
        "probability": probability,
        "predicted_result": predicted_result
    }
    _append_jsonl_record(CONTRIBUTIONS_FILE, contribution_record)

    # Phase 2: Uncertainty interval judgment [0.3, 0.7]
    is_uncertain = 0.3 <= probability <= 0.7
    review_flag = False

    if is_uncertain:
        review_flag = True
        queue_item = {
            "id": f"UNC_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            "timestamp": datetime.now().isoformat(),
            "file_name": file_name,
            "model_type": model_type,
            "probability": round(probability, 4),
            "predicted_result": predicted_result,
            "user_name": user_name.strip() if user_name else "Anonymous Contributor",
            "user_email": user_email.strip() if user_email else "",
            "status": "Pending Manual Review",
            "peaks_count": len(peaks),
            "peaks": peaks
        }
        _append_jsonl_record(UNCERTAIN_QUEUE_FILE, queue_item)
        status_msg = "Authorized: Sample probability falls in uncertainty interval [0.3, 0.7]. Saved to pending review queue."
    else:
        status_msg = "Authorized: Sample confidence is high. Results archived for model improvement."

    return {
        "authorized": True,
        "status": status_msg,
        "is_uncertain": is_uncertain,
        "review_flag": review_flag
    }


def get_queue_stats() -> Dict[str, int]:
    """Get review queue and contributions statistics with O(1) memory overhead."""
    return {
        "total_uncertain": _count_jsonl_lines(UNCERTAIN_QUEUE_FILE),
        "total_contributions": _count_jsonl_lines(CONTRIBUTIONS_FILE)
    }
