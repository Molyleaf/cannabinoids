import os
import re
import csv
import random
import math
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

# ==================== Global Styles: Calibri + Underline Input ====================
display(HTML("""
<style>
    .widget-label, .widget-text, .widget-float, .widget-int, .widget-button,
    .jupyter-widgets {
        font-family: 'Calibri', 'Arial', sans-serif !important;
        font-size: 18px !important;
    }
    .widget-label {
        font-size: 18px !important;
    }
    .widget-button {
        font-size: 18px !important;
    }
    .widget-text input, .widget-float input, .widget-int input {
        border: none !important;
        border-bottom: 1.5px solid #999 !important;
        border-radius: 0 !important;
        background: transparent !important;
        outline: none !important;
        box-shadow: none !important;
        font-family: 'Calibri', 'Arial', sans-serif !important;
        font-size: 18px !important;
        padding: 2px 4px !important;
        margin: 0 !important;
    }
    .widget-text input:focus, .widget-float input:focus, .widget-int input:focus {
        border-bottom: 1.5px solid #2c7fb8 !important;
    }
    .output_area, .output_area pre, .output_area .output_text,
    .output_area .stream, .output_area .stdout, .output_area .stderr {
        font-family: 'Calibri', 'Arial', sans-serif !important;
        font-size: 18px !important;
    }
    .main-title {
        font-family: 'Calibri', 'Arial', sans-serif !important;
        font-size: 28px !important;
        font-weight: bold !important;
    }
    .section-title {
        font-family: 'Calibri', 'Arial', sans-serif !important;
        font-size: 18px !important;
        font-weight: bold !important;
    }
</style>
"""))

# ==================== Dependency Check ====================
try:
    from rdkit import Chem
    from rdkit.Chem import DataStructs
    from rdkit.Chem.Fingerprints import FingerprintMols
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("[WARN] rdkit not installed. Structure similarity will be disabled.")


# ==================== Data Classes ====================
class SpectraEntry:
    def __init__(self):
        self.name = ""
        self.mw = 0.0
        self.smiles = ""
        self.peaks = []

    def to_msp_string(self) -> str:
        if not self.peaks:
            return ""
        lines = [f"Name: {self.name}"]
        lines.append(f"Num peaks: {len(self.peaks)}")
        for i in range(0, len(self.peaks), 5):
            chunk = self.peaks[i:i+5]
            line = "  " + " ".join([f"{int(mz):>4} {int(intensity):>4};" for mz, intensity in chunk])
            lines.append(line)
        return "\n".join(lines) + "\n\n"


class MSPParser:
    @staticmethod
    def parse_file(file_path: str) -> List[SpectraEntry]:
        if not os.path.exists(file_path):
            return []
        entries = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        parts = content.split('Name:')
        for part in parts[1:]:
            if not part.strip():
                continue
            lines = part.strip().splitlines()
            if not lines:
                continue
            entry = SpectraEntry()
            entry.name = lines[0].strip()
            peaks = []
            peak_lines = []
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('MW:'):
                    parts_mw = line.split()
                    for i, token in enumerate(parts_mw):
                        if token == 'MW:' and i+1 < len(parts_mw):
                            try:
                                entry.mw = float(parts_mw[i+1])
                            except:
                                pass
                elif line.startswith('SMILES:'):
                    entry.smiles = line.split(':', 1)[1].strip()
                elif line.startswith('Num peaks:'):
                    continue
                elif re.match(r'^\s*\d+\s+\d+', line) or re.match(r'^\s*\d+\s+Tr', line):
                    peak_lines.append(line)
            if not peak_lines:
                idx = -1
                for i, line in enumerate(lines):
                    if 'm/z Values and Intensities:' in line:
                        idx = i + 1
                        break
                if idx != -1:
                    for line in lines[idx:]:
                        if '|' in line or (re.match(r'^\s*\d+\s+\d+', line.strip())):
                            peak_lines.append(line)
                        else:
                            break
            if peak_lines:
                all_text = ' '.join(peak_lines)
                pattern = re.compile(r'(\d+)\s+(Tr|\d+\.?\d*)')
                for match in pattern.finditer(all_text):
                    mz = int(match.group(1))
                    int_str = match.group(2)
                    intensity = 1.0 if int_str.lower() == 'tr' else float(int_str)
                    peaks.append((mz, intensity))
            entry.peaks = peaks
            if len(peaks) >= 5:
                entries.append(entry)
        return entries


# ==================== Core Library Builder ====================
class NegativeLibraryBuilder:
    def __init__(self, target_category: str, target_smiles: str, target_mw: float, target_count: int):
        self.target_category = target_category
        self.target_smiles = target_smiles
        self.target_mw = target_mw
        self.target_count = target_count
        self.selected_entries = []
        self.matrix_entries = []
        self.nps_entries = []
        self.mona_entries = []

    def log(self, msg: str):
        print(msg)

    def load_libraries(self):
        self.log("\n[INFO] Loading library files...")
        matrix_path = 'Matrix.msp'
        if os.path.exists(matrix_path):
            self.matrix_entries = MSPParser.parse_file(matrix_path)
            self.log(f"  Matrix: {len(self.matrix_entries)} compounds")
        else:
            self.log(f"  Warning: Matrix.msp not found")
        nps_path = 'NPS.msp'
        if os.path.exists(nps_path):
            self.nps_entries = MSPParser.parse_file(nps_path)
            self.log(f"  NPS: {len(self.nps_entries)} compounds")
        else:
            self.log(f"  Warning: NPS.msp not found")
        mona_path = 'MONA.msp'
        if os.path.exists(mona_path):
            self.mona_entries = MSPParser.parse_file(mona_path)
            self.log(f"  MONA: {len(self.mona_entries)} compounds")
        else:
            self.log(f"  Warning: MONA.msp not found")

    def find_csv_file(self) -> Optional[str]:
        neg_dir = 'Neg'
        if not os.path.exists(neg_dir):
            return None
        target_norm = self.target_category.strip().lower()
        if target_norm.endswith('s') and not target_norm.endswith('ss'):
            target_singular = target_norm[:-1]
        else:
            target_singular = target_norm
        csv_files = [f for f in os.listdir(neg_dir) if f.endswith('.csv')]
        for csv_file in csv_files:
            if csv_file.startswith('cayman_product_names_'):
                name_part = csv_file.replace('cayman_product_names_', '').replace('.csv', '')
                name_part_norm = name_part.strip().lower()
                if target_norm == name_part_norm or target_singular == name_part_norm:
                    return os.path.join(neg_dir, csv_file)
                if name_part_norm.endswith('s') and not name_part_norm.endswith('ss'):
                    if target_norm == name_part_norm[:-1]:
                        return os.path.join(neg_dir, csv_file)
        for csv_file in csv_files:
            if csv_file.startswith('cayman_product_names_'):
                name_part = csv_file.replace('cayman_product_names_', '').replace('.csv', '')
                name_part_norm = name_part.strip().lower()
                if target_norm in name_part_norm or name_part_norm in target_norm:
                    return os.path.join(neg_dir, csv_file)
        return None

    def load_exclusion_names(self) -> Set[str]:
        csv_path = self.find_csv_file()
        if not csv_path:
            return set()
        exclude_names = set()
        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                name_col = None
                for col in (reader.fieldnames or []):
                    if col.lower() in ['name', 'compound', 'compound_name', 'chemical']:
                        name_col = col
                        break
                if name_col is None and reader.fieldnames:
                    name_col = reader.fieldnames[0]
                if name_col is None:
                    return exclude_names
                for row in reader:
                    val = row.get(name_col, '').strip()
                    if val:
                        val = val.strip('"\'')
                        exclude_names.add(val)
            self.log(f"  Loaded {len(exclude_names)} exclusion names from {os.path.basename(csv_path)}")
        except Exception as e:
            self.log(f"  [ERROR] Failed to read CSV: {e}")
        return exclude_names

    def is_excluded(self, entry: SpectraEntry, exclude_names: Set[str]) -> bool:
        if not exclude_names:
            return False
        entry_name = entry.name.strip().lower().replace(' ', '')
        entry_name = entry_name.replace('-', '').replace('_', '')
        for ex in exclude_names:
            ex_clean = ex.strip().lower().replace(' ', '')
            ex_clean = ex_clean.replace('-', '').replace('_', '')
            if not ex_clean:
                continue
            if ex_clean in entry_name or entry_name in ex_clean:
                return True
        return False

    def calculate_similarity(self, smiles1: str, smiles2: str) -> float:
        if not RDKIT_AVAILABLE or not smiles1 or not smiles2:
            return 0.0
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            if mol1 is None or mol2 is None:
                return 0.0
            fp1 = FingerprintMols.FingerprintMol(mol1)
            fp2 = FingerprintMols.FingerprintMol(mol2)
            return DataStructs.FingerprintSimilarity(fp1, fp2)
        except:
            return 0.0

    def compute_combined_score(self, entry: SpectraEntry) -> float:
        """
        Compute combined score: 50% MW matching + 50% structural similarity
        """
        if entry.mw <= 0:
            return 0.0
        
        # MW score: Gaussian function, sigma=30
        mw_score = math.exp(-((entry.mw - self.target_mw) ** 2) / (2 * 30 ** 2))
        
        # Structural similarity score
        sim_score = self.calculate_similarity(self.target_smiles, entry.smiles)
        
        # 50% each
        combined_score = 0.5 * mw_score + 0.5 * sim_score
        return combined_score

    def sample_with_expanding_mw_window(self, candidates: List[SpectraEntry],
                                         target_count: int,
                                         initial_window: int = 20,
                                         max_window: int = 500,
                                         step: int = 25) -> List[SpectraEntry]:
        """
        Sample using a progressively expanding molecular weight window.
        The window can expand infinitely up to max_window.
        
        Args:
            candidates: List of candidate compounds
            target_count: Target number of samples
            initial_window: Initial MW window (±20 Da)
            max_window: Maximum MW window (±500 Da)
            step: Step size for each expansion (25 Da)
        """
        if not candidates or target_count <= 0:
            return []
        
        # Sort by combined score (50% MW + 50% similarity)
        scored = []
        for entry in candidates:
            score = self.compute_combined_score(entry)
            scored.append((entry, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Gradually expand window
        window = initial_window
        while window <= max_window:
            lower = self.target_mw - window
            upper = self.target_mw + window
            
            # Filter candidates within current window
            window_candidates = [(e, s) for e, s in scored if lower <= e.mw <= upper]
            
            if len(window_candidates) >= target_count:
                # Sample from window candidates by combined score
                selected = random.sample([e for e, _ in window_candidates[:target_count * 3]],
                                          min(target_count, len(window_candidates)))
                self.log(f"  Sampled {len(selected)} compounds within ±{window} Da window")
                return selected
            
            # Current window insufficient, expand
            window += step
            self.log(f"  Expanding MW window to ±{window} Da...")
        
        # If max window reached and still insufficient, return all available
        self.log(f"  Warning: Only {len(scored)} compounds available within ±{max_window} Da")
        return [e for e, _ in scored[:target_count]]

    def build_library(self):
        self.load_libraries()
        exclude_names = self.load_exclusion_names()
        
        # Step 1: Include all Matrix compounds
        selected = list(self.matrix_entries)
        self.log(f"\n[Step 1] Include Matrix: {len(selected)} compounds")
        
        current_count = len(selected)
        remaining = self.target_count - current_count
        if remaining <= 0:
            self.log("[WARN] Matrix already meets target count")
            self.selected_entries = selected[:self.target_count]
            return
        
        self.log(f"\n[Target] Need {remaining} more compounds")
        self.log(f"  Target MW: {self.target_mw}")
        self.log(f"  MW weight: 50%, Similarity weight: 50%")
        
        # Step 2: NPS sampling (after excluding target category, using combined score + expanding window)
        self.log("\n[Step 2] NPS sampling (50% MW + 50% similarity)...")
        nps_candidates = []
        for entry in self.nps_entries:
            if self.is_excluded(entry, exclude_names):
                continue
            if entry.mw <= 0:
                continue
            nps_candidates.append(entry)
        
        self.log(f"  NPS candidates (with MW): {len(nps_candidates)}")
        
        # NPS and MONA each contribute half
        nps_target = remaining // 2
        mona_target = remaining - nps_target
        
        if nps_candidates and nps_target > 0:
            sampled_nps = self.sample_with_expanding_mw_window(
                nps_candidates, nps_target, initial_window=20, max_window=500, step=25
            )
            selected.extend(sampled_nps)
            self.log(f"  NPS sampled: {len(sampled_nps)} compounds")
        
        current_count = len(selected)
        remaining_after_nps = self.target_count - current_count
        self.log(f"  Cumulative: {current_count}, remaining: {remaining_after_nps}")
        
        # Step 3: MONA sampling (using combined score + expanding window)
        self.log("\n[Step 3] MONA sampling (50% MW + 50% similarity)...")
        if self.mona_entries and remaining_after_nps > 0:
            mona_candidates = []
            for entry in self.mona_entries:
                if entry.mw <= 0:
                    continue
                mona_candidates.append(entry)
            
            self.log(f"  MONA candidates (with MW): {len(mona_candidates)}")
            
            sampled_mona = self.sample_with_expanding_mw_window(
                mona_candidates, remaining_after_nps, initial_window=20, max_window=500, step=25
            )
            selected.extend(sampled_mona)
            self.log(f"  MONA sampled: {len(sampled_mona)} compounds")
        
        # Final trimming
        if len(selected) > self.target_count:
            selected = random.sample(selected, self.target_count)
        elif len(selected) < self.target_count:
            self.log(f"\n[WARN] Only {len(selected)} compounds available, target {self.target_count}")
        
        # Final statistics
        final_mws = [e.mw for e in selected if e.mw > 0]
        if final_mws:
            final_avg = np.mean(final_mws)
            self.log(f"\n[Final Statistics]")
            self.log(f"  Total compounds: {len(selected)}")
            self.log(f"  Average MW: {final_avg:.1f} (target: {self.target_mw:.1f})")
            self.log(f"  MW deviation: {final_avg - self.target_mw:+.1f}")
            self.log(f"  MW range: {min(final_mws):.1f} - {max(final_mws):.1f}")
        
        self.selected_entries = selected

    def save_msp(self, output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in self.selected_entries:
                f.write(entry.to_msp_string())
        self.log(f"\n[Saved] {len(self.selected_entries)} compounds written to {output_path}")


# ==================== UI ====================
def create_labeled_input(label_text, input_widget, width='500px'):
    label = widgets.HTML(
        f"<span style='font-family: Calibri, Arial, sans-serif; font-size: 18px; font-weight: normal; margin-right: 0px; padding-right: 0px;'>{label_text}</span>"
    )
    input_widget.layout.width = width
    return widgets.HBox([label, input_widget], layout=widgets.Layout(
        align_items='center',
        justify_content='flex-start',
        gap='0px',
        margin='0px',
        padding='0px',
        width='100%'
    ))


def build_library_callback(target_category, target_smiles, target_mw, target_count,
                           output, seed, output_area):
    random.seed(seed)
    np.random.seed(seed)
    output_area.clear_output()
    with output_area:
        print("="*60)
        print("Negative Library Builder")
        print("="*60)
        print(f"Target Category: {target_category}")
        print(f"Target MW: {target_mw}")
        print(f"Target Count: {target_count}")
        print(f"Output File: {output}")
        print("="*60)
        builder = NegativeLibraryBuilder(
            target_category=target_category,
            target_smiles=target_smiles,
            target_mw=target_mw,
            target_count=target_count
        )
        builder.build_library()
        builder.save_msp(output)
        print("\n" + "="*60)
        print("Build completed!")
        print("="*60)


def create_ui():
    category_input = widgets.Text(value='Cathinone', placeholder='e.g., Cathinone')
    smiles_input = widgets.Text(value='C[C@@H](C(=O)C1=CC=CC=C1)N',
                                placeholder='Enter target compound SMILES')
    mw_input = widgets.FloatText(value=230.0, placeholder='e.g., 230.0')
    count_input = widgets.IntText(value=2000, placeholder='e.g., 2000')
    output_input = widgets.Text(value='negative_library.msp', placeholder='Output MSP filename')
    seed_input = widgets.IntText(value=42, placeholder='e.g., 42')

    row_category = create_labeled_input('Target Category:', category_input, '500px')
    row_smiles = create_labeled_input('Target SMILES:', smiles_input, '500px')
    row_mw = create_labeled_input('Target MW:', mw_input, '500px')
    row_count = create_labeled_input('Target Count:', count_input, '500px')
    row_output = create_labeled_input('Output File:', output_input, '500px')
    row_seed = create_labeled_input('Random Seed:', seed_input, '500px')

    run_button = widgets.Button(
        description='Build Library',
        button_style='success',
        layout=widgets.Layout(width='220px')
    )

    output_area = widgets.Output()

    def on_run_clicked(b):
        build_library_callback(
            target_category=category_input.value,
            target_smiles=smiles_input.value,
            target_mw=mw_input.value,
            target_count=count_input.value,
            output=output_input.value,
            seed=seed_input.value,
            output_area=output_area
        )

    run_button.on_click(on_run_clicked)

    ui = widgets.VBox([
        widgets.HTML("<div class='main-title'>Negative Library Builder</div>"),
        widgets.HTML("<hr>"),
        widgets.HTML("<span class='section-title'>Required files:</span> Matrix.msp, NPS.msp, MONA.msp"),
        widgets.HTML("<span class='section-title'>CSV files:</span> Neg/cayman_product_names_*.csv"),
        widgets.HTML("<hr>"),
        row_category,
        row_smiles,
        row_mw,
        row_count,
        row_output,
        row_seed,
        widgets.HBox([run_button], layout=widgets.Layout(
            justify_content='flex-start',
            gap='0px',
            margin='0px',
            padding='0px',
            width='100%'
        )),
        output_area
    ], layout=widgets.Layout(
        align_items='flex-start',
        gap='2px',
        margin='0px',
        padding='0px',
        width='100%'
    ))

    return ui


ui = create_ui()
display(ui)