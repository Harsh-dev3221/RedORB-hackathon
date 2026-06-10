"""Benchmark embedding throughput to size the full 1M-sentence precompute."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from verdict.embedder import embed_passages, get_model

texts = [
    f"Built and maintained data pipelines on Apache Airflow processing {i} GB of "
    "daily transactional data across source systems, working with Spark for batch "
    "processing and dbt for the transformation layer in our Snowflake warehouse."
    for i in range(2000)
]

if __name__ == "__main__":
    big = texts * 4  # 8000 sentences so process spin-up amortizes
    for parallel in (4, 8):
        model = get_model(threads=2)
        t0 = time.time()
        embed_passages(model, big, batch_size=512, parallel=parallel)
        dt = time.time() - t0
        print(f"parallel={parallel}: {len(big)/dt:.0f} sentences/s ({dt:.1f}s for {len(big)})")
        del model
