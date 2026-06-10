"""Find the embedding bottleneck: text length vs batch size, on real data."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from verdict.data import iter_candidates
from verdict.embedder import embed_passages, get_model
from verdict.evidence import build_record

sample = sys.argv[1]
real: list[str] = []
for c in iter_candidates(sample):
    real.extend(build_record(c).sentences)
real = (real * 40)[:4000]  # ~4000 real-length sentences
short = ["Built a search ranking system for users."] * 4000

model = get_model()
embed_passages(model, real[:32])  # warmup

for name, data in (("real", real), ("short", short)):
    for bs in (32, 128, 512):
        t0 = time.time()
        embed_passages(model, data, batch_size=bs)
        dt = time.time() - t0
        print(f"{name:>5} bs={bs:<4} {len(data)/dt:7.0f} sentences/s")
