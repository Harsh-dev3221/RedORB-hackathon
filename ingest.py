"""Incremental ingest: add new candidates to the live index WITHOUT a full rebuild.

Embeds only the new profiles and appends to the existing artifacts in place
(atomic tmp-file replace), so search.py / rank.py / explain.py see them
immediately with zero loader changes.

  python ingest.py --input new_candidates.jsonl            # JSON/JSONL/JSONL.GZ
  python ingest.py --input batch.jsonl --replace           # upsert existing ids
  python ingest.py --input resume.txt --format text        # raw resume via Gemini

New candidates may lack redrob_signals (external uploads): every downstream
engine treats missing signals as unknown-mild-prior, never as zero.
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
import time
from pathlib import Path

import numpy as np
import orjson

sys.path.insert(0, str(Path(__file__).parent / "src"))

from verdict.data import iter_candidates
from verdict.embedder import DIM, embed_passages, get_model
from verdict.evidence import build_record, record_from_dict, record_to_dict

ART = Path(__file__).parent / "artifacts"


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    # tmp name must end in .npy or np.save silently appends the extension
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def load_index():
    ids = (ART / "candidate_ids.txt").read_text(encoding="utf-8").splitlines()
    counts = np.load(ART / "sent_counts.npy")
    vecs = np.load(ART / "evidence_vectors.npy")
    means = np.load(ART / "mean_vecs.npy")
    return ids, counts, vecs, means


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--format", choices=["json", "text"], default="json",
                    help="json: candidate-schema JSON/JSONL; text: raw resume via Gemini")
    ap.add_argument("--replace", action="store_true",
                    help="upsert: replace candidates whose ids already exist")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate + twin-check the upload without writing anything")
    ap.add_argument("--strict", action="store_true",
                    help="reject the whole upload if any twin is detected")
    ap.add_argument("--skip-twins", action="store_true",
                    help="skip near-duplicate detection (faster for trusted bulk loads)")
    args = ap.parse_args()
    t0 = time.time()

    # ---- read input ----
    if args.format == "text":
        from verdict.resume_parser import parse_resume_text

        raw = Path(args.input).read_text(encoding="utf-8")
        new_cands = [parse_resume_text(raw)]
        print(f"Gemini parsed resume -> candidate {new_cands[0]['candidate_id']}")
    else:
        new_cands = list(iter_candidates(args.input))
    if not new_cands:
        raise SystemExit("no candidates in input")

    # ---- validate ids ----
    seen_new: set[str] = set()
    for c in new_cands:
        cid = c.get("candidate_id", "")
        if not cid:
            raise SystemExit("candidate without candidate_id")
        if cid in seen_new:
            raise SystemExit(f"duplicate candidate_id inside upload: {cid}")
        seen_new.add(cid)

    ids, counts, vecs, means = load_index()
    existing = set(ids)
    dups = seen_new & existing
    if dups and not args.replace:
        raise SystemExit(
            f"{len(dups)} ids already indexed (e.g. {sorted(dups)[:3]}); rerun with --replace to upsert"
        )

    # ---- build records + embed only the new sentences ----
    new_records = [build_record(c) for c in new_cands]
    new_sentences: list[str] = []
    new_counts: list[int] = []
    for rec in new_records:
        new_counts.append(len(rec.sentences))
        new_sentences.extend(rec.sentences)
    model = get_model(threads=args.threads, cuda=args.cuda)
    new_vecs = (
        embed_passages(model, new_sentences, batch_size=32).astype(np.float16)
        if new_sentences
        else np.zeros((0, DIM), dtype=np.float16)
    )
    new_means = np.zeros((len(new_records), DIM), dtype=np.float16)
    off = 0
    for i, k in enumerate(new_counts):
        if k:
            m = new_vecs[off : off + k].astype(np.float32).mean(axis=0)
            n = np.linalg.norm(m)
            if n > 1e-9:
                new_means[i] = (m / n).astype(np.float16)
        off += k
    print(f"embedded {len(new_sentences)} sentences for {len(new_records)} candidates")

    # ---- upload-time twin detection (fraud-resistance at the door) ----
    n_twins = 0
    if not args.skip_twins:
        from verdict.twins import find_twins

        index_records = []
        with gzip.open(ART / "records.jsonl.gz", "rb") as f:
            for line in f:
                if line.strip():
                    index_records.append(record_from_dict(orjson.loads(line)))
        twin_flags = find_twins(new_records, new_means, ids, means, index_records)
        for rec, tf in zip(new_records, twin_flags):
            if not tf:
                continue
            n_twins += 1
            for t in tf:
                # lands in the stored ledger -> credibility engine dings it at rank time
                if not (args.replace and rec.candidate_id in dups
                        and rec.candidate_id in t):
                    rec.suspicions.append(t)
                print(f"  TWIN {rec.candidate_id}: {t}")
        if n_twins and args.strict:
            raise SystemExit(f"strict mode: rejected upload - {n_twins} twin candidate(s)")
    if args.dry_run:
        print(f"dry run: {len(new_records)} candidates validated, {n_twins} twin(s); nothing written")
        return

    # ---- drop superseded rows on upsert ----
    if dups:
        offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        keep = [i for i, cid in enumerate(ids) if cid not in dups]
        sent_mask = np.zeros(len(vecs), dtype=bool)
        for i in keep:
            sent_mask[offsets[i] : offsets[i + 1]] = True
        vecs = vecs[sent_mask]
        counts = counts[keep]
        means = means[keep]
        ids = [ids[i] for i in keep]
        print(f"upsert: dropped {len(dups)} superseded candidates")

    # ---- append + atomic replace ----
    _atomic_save_npy(ART / "evidence_vectors.npy", np.concatenate([vecs, new_vecs]))
    _atomic_save_npy(ART / "sent_counts.npy",
                     np.concatenate([counts, np.asarray(new_counts, dtype=np.int32)]))
    _atomic_save_npy(ART / "mean_vecs.npy", np.concatenate([means, new_means]))
    tmp_ids = ART / "candidate_ids.txt.tmp"
    tmp_ids.write_text("\n".join(ids + [r.candidate_id for r in new_records]), encoding="utf-8")
    os.replace(tmp_ids, ART / "candidate_ids.txt")

    # records.jsonl.gz: append as a new gzip member (multi-member reads are transparent);
    # on upsert we must rewrite to drop superseded rows
    rec_path = ART / "records.jsonl.gz"
    if dups:
        kept_lines = []
        with gzip.open(rec_path, "rb") as f:
            for line in f:
                if line.strip() and orjson.loads(line)["candidate_id"] not in dups:
                    kept_lines.append(line)
        tmp = rec_path.with_suffix(".gz.tmp")
        with gzip.open(tmp, "wb") as f:
            f.writelines(kept_lines)
            for rec in new_records:
                f.write(orjson.dumps(record_to_dict(rec)) + b"\n")
        os.replace(tmp, rec_path)
    else:
        with gzip.open(rec_path, "ab") as f:
            for rec in new_records:
                f.write(orjson.dumps(record_to_dict(rec)) + b"\n")

    total = len(ids) + len(new_records)
    print(f"index now {total} candidates (+{len(new_records)}) in {time.time()-t0:.1f}s - searchable immediately")


if __name__ == "__main__":
    main()
