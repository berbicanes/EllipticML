"""Project-wide constants and paths.

Every experiment imports its settings from here so that the split, the seeds
and the file locations are defined exactly once.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = Path(os.environ.get("ELLIPTIC_DATA_DIR", ROOT / "data" / "raw"))
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"
ARTIFACTS = ROOT / "artifacts"

FEATURES_CSV = "elliptic_txs_features.csv"
CLASSES_CSV = "elliptic_txs_classes.csv"
EDGES_CSV = "elliptic_txs_edgelist.csv"

# Dataset layout: txId, time step, 93 local features, 72 aggregated features.
N_LOCAL = 93
N_AGG = 72
N_STEPS = 49

# Labels after encoding. Unknown rows keep -1 and are never used for supervised
# training or evaluation.
ILLICIT = 1
LICIT = 0
UNKNOWN = -1

# Temporal split from Weber et al. (2019): steps 1-34 train, 35-49 test.
TRAIN_STEPS = range(1, 35)
TEST_STEPS = range(35, 50)

# Inside the training period, the last steps are held out for model selection
# and threshold tuning (never the test steps).
VALIDATION_STEPS = range(30, 35)

# The dark market shutdown reported by Weber et al. happens at step 43.
DARK_MARKET_STEP = 43

SEEDS = (0, 1, 2, 3, 4)
DEFAULT_SEED = 0
