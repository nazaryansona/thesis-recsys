from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "MOOCCube"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

SEED = 42
K_VALUES = [5, 10, 20]
MIN_COURSES_PER_USER = 3