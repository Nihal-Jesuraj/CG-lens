"""Write a synthetic Cheng-like dataset for smoke tests (NOT scientific data)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.synthetic import make_synthetic_dataset  # noqa: E402

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_raw")
    info = make_synthetic_dataset(out)
    print(f"wrote {len(info['files'])} synthetic files to {out}")
