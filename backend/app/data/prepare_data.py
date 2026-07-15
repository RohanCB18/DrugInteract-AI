"""
prepare_data.py — Download and preprocess DDI + unlabeled molecule data.

Usage:
    python -m backend.app.data.prepare_data

Downloads datasets DIRECTLY from Harvard Dataverse (no PyTDC required):
  1. DrugBank DDI  (file ID 4139573)  → labeled drug pairs + interaction types
  2. ZINC-250K     (file ID 4170963)  → unlabeled SMILES for self-supervised pretraining

Bins the ~86 DrugBank interaction types into 5 major categories and saves
processed splits to disk.
"""

import json
import pickle
import sys
import io
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from backend.app.config import settings


# ══════════════════════════════════════════════════════════════════════════════
#  Interaction-type binning
# ══════════════════════════════════════════════════════════════════════════════

BIN_RULES = {
    0: {  # "Increases effect / toxicity"
        "keywords": [
            "increase", "enhance", "potentiate", "synerg",
            "raise", "elevat", "higher", "amplif",
        ],
    },
    1: {  # "Decreases effect"
        "keywords": [
            "decrease", "reduce", "diminish", "lower",
            "attenuate", "weaken", "less effective", "inhibit",
        ],
    },
    2: {  # "Alters metabolism / pharmacokinetics"
        "keywords": [
            "metabolism", "metabol", "clearance", "absorption",
            "bioavailability", "excretion", "half-life", "cyp",
            "serum", "concentration", "level", "exposure",
        ],
    },
    3: {  # "Increases adverse / risk"
        "keywords": [
            "risk", "adverse", "toxic", "side effect",
            "arrhythmia", "bleed", "serotonin", "seizure",
            "hypotension", "cns depression", "qt", "nephro",
            "hepato", "cardio", "neuro",
        ],
    },
    # 4 → catch-all "Other interaction"
}

BIN_LABELS = {
    0: "Increases effect",
    1: "Decreases effect",
    2: "Alters metabolism",
    3: "Increases risk",
    4: "Other interaction",
}

# Harvard Dataverse file IDs (from TDC repository: doi:10.7910/DVN/21LKWG)
DATAVERSE_BASE = "https://dataverse.harvard.edu/api/access/datafile"
DRUGBANK_FILE_ID = 4139573
ZINC_FILE_ID = 4170963


def _bin_interaction(description: str) -> int:
    """Map a free-text interaction description to one of 5 bins."""
    desc_lower = str(description).lower()
    for bin_id, rule in BIN_RULES.items():
        for kw in rule["keywords"]:
            if kw in desc_lower:
                return bin_id
    return 4  # "Other"


def _download_dataverse(file_id: int, description: str) -> bytes:
    """Download a file from Harvard Dataverse by file ID."""
    url = f"{DATAVERSE_BASE}/{file_id}"
    print(f"  Downloading {description} from Dataverse (file ID: {file_id})...")
    print(f"  URL: {url}")

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; DrugInteractAI/1.0)",
            "Accept": "*/*",
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        print(f"  Downloaded {len(data) / 1024 / 1024:.1f} MB")
        return data
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Failed to download {description} from Dataverse (HTTP {e.code}). "
            f"The Dataverse server may require browser-like session cookies for large files. "
            f"See MANUAL_DOWNLOAD.md for instructions."
        ) from e


def download_ddi_data(data_dir: Path) -> pd.DataFrame:
    """
    Download DrugBank DDI dataset from Harvard Dataverse.

    Returns DataFrame with columns:
        drug_a_id, drug_b_id, smiles_a, smiles_b, interaction_raw
    """
    cache_path = data_dir / "raw" / "drugbank.tab"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"  Using cached DrugBank DDI at {cache_path}")
    else:
        raw = _download_dataverse(DRUGBANK_FILE_ID, "DrugBank DDI")
        cache_path.write_bytes(raw)

    df = pd.read_csv(cache_path, sep="\t")
    print(f"  Columns found: {df.columns.tolist()}")

    # Handle column name conventions from the Dataverse .tab file:
    #   ID1 / ID2  → drug IDs
    #   X1  / X2   → SMILES strings
    #   Y          → interaction label / description
    #   Map        → original fine-grained label (kept for reference)
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ("id1", "drug1_id", "drug_1_id"):
            col_map[col] = "drug_a_id"
        elif cl in ("id2", "drug2_id", "drug_2_id"):
            col_map[col] = "drug_b_id"
        elif cl in ("x1", "drug1"):
            col_map[col] = "smiles_a"
        elif cl in ("x2", "drug2"):
            col_map[col] = "smiles_b"
        elif cl == "y":
            col_map[col] = "interaction_raw"

    df = df.rename(columns=col_map)

    # Validate required columns
    required = {"smiles_a", "smiles_b", "interaction_raw"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"DrugBank DDI file is missing expected columns: {missing}. "
            f"Got columns: {df.columns.tolist()}"
        )

    # Drop rows with missing SMILES
    before = len(df)
    df = df.dropna(subset=["smiles_a", "smiles_b"]).reset_index(drop=True)
    print(f"  Loaded {len(df)} pairs (dropped {before - len(df)} with missing SMILES)")
    return df


def download_zinc250k(data_dir: Path) -> list:
    """
    Download ZINC-250K unlabeled SMILES from Harvard Dataverse.
    Returns a list of SMILES strings.
    """
    cache_path = data_dir / "raw" / "zinc.tab"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"  Using cached ZINC at {cache_path}")
    else:
        raw = _download_dataverse(ZINC_FILE_ID, "ZINC-250K")
        cache_path.write_bytes(raw)

    df = pd.read_csv(cache_path, sep="\t")
    print(f"  ZINC columns: {df.columns.tolist()}")

    # Find the SMILES column (named 'smiles', 'SMILES', or similar)
    smiles_col = None
    for col in df.columns:
        if col.lower() == "smiles":
            smiles_col = col
            break

    if smiles_col is None:
        # Fall back to first column
        smiles_col = df.columns[0]
        print(f"  Warning: no 'smiles' column found, using '{smiles_col}'")

    smiles_list = df[smiles_col].dropna().tolist()
    print(f"  Loaded {len(smiles_list)} ZINC molecules")
    return smiles_list


# ══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def prepare_all(data_dir: Path = None):
    """Run the full data preparation pipeline."""
    if data_dir is None:
        data_dir = settings.DATA_DIR

    output_dir = data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. DDI data ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 1: Downloading DrugBank DDI dataset")
    print("=" * 60)
    df = download_ddi_data(data_dir)

    # ── 2. Bin interactions ───────────────────────────────────────────────
    print("\n  Step 2: Binning interaction types into 5 categories")

    if df["interaction_raw"].dtype == object:
        # Text descriptions → bin them
        df["label"] = df["interaction_raw"].apply(_bin_interaction)
    else:
        # Already numeric (integer class index) → map to 5 bins modularly
        unique_labels = sorted(df["interaction_raw"].unique())
        num_original = len(unique_labels)
        print(f"  Found {num_original} original classes, mapping to 5 bins...")
        label_map = {orig: i % 5 for i, orig in enumerate(unique_labels)}
        df["label"] = df["interaction_raw"].map(label_map)

    label_counts = Counter(df["label"].tolist())
    print("  Label distribution:")
    for lbl, count in sorted(label_counts.items()):
        print(f"    {lbl} ({BIN_LABELS.get(lbl, '?')}): {count}")

    # ── 3. Split ──────────────────────────────────────────────────────────
    print("\n  Step 3: Train/val/test split (70/10/20)")
    np.random.seed(42)
    n = len(df)
    indices = np.random.permutation(n)
    train_end = int(0.7 * n)
    val_end = int(0.8 * n)

    splits = {
        "train": df.iloc[indices[:train_end]].reset_index(drop=True),
        "val":   df.iloc[indices[train_end:val_end]].reset_index(drop=True),
        "test":  df.iloc[indices[val_end:]].reset_index(drop=True),
    }

    for name, split_df in splits.items():
        print(f"    {name}: {len(split_df)} pairs")
        split_df.to_pickle(output_dir / f"ddi_{name}.pkl")

    # Save label map
    with open(output_dir / "label_map.json", "w") as f:
        json.dump(BIN_LABELS, f, indent=2)

    # Save drug list for frontend autocomplete
    all_drugs = []
    seen = set()
    for _, row in df.iterrows():
        for smi, did in [(row["smiles_a"], row.get("drug_a_id", "")),
                         (row["smiles_b"], row.get("drug_b_id", ""))]:
            if smi and isinstance(smi, str) and smi not in seen:
                all_drugs.append({"smiles": smi, "id": str(did)})
                seen.add(smi)

    with open(output_dir / "drug_list.json", "w") as f:
        json.dump(all_drugs, f)

    print(f"\n  Saved {len(all_drugs)} unique drugs for frontend autocomplete.")

    # ── 4. Unlabeled ZINC data ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 4: Downloading ZINC-250K for pretraining")
    print("=" * 60)
    zinc_smiles = download_zinc250k(data_dir)

    with open(output_dir / "zinc_smiles.pkl", "wb") as f:
        pickle.dump(zinc_smiles, f)

    print(f"\n  Saved {len(zinc_smiles)} ZINC SMILES.")

    # ── Done ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ Data preparation complete!")
    print(f"  Output directory: {output_dir}")
    print("=" * 60 + "\n")

    return splits, zinc_smiles


if __name__ == "__main__":
    prepare_all()

