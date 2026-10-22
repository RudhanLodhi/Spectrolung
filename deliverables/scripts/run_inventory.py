from argparse import ArgumentParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inventory import build_inventory, write_report

parser = ArgumentParser()
parser.add_argument("--data-root", type=Path, default=Path("HF_Lung"))
parser.add_argument("--output", type=Path, default=ROOT / "results" / "00_inventory.md")
args = parser.parse_args()
write_report(build_inventory(args.data_root), args.output)
print(args.output)
