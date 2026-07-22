"""Search the G-Mixup augmented ratio on DrugOOD IC50."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


AUGMENTED_RATIOS = (0.15, 0.25, 0.5)


def value_name(value):
    return f"{value:g}".replace("-", "m").replace(".", "p")


def stats(values):
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"mean": None, "std": None}
    return {"mean": statistics.fmean(finite), "std": statistics.pstdev(finite)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="+", choices=("assay", "scaffold", "size"), default=["assay"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--subset", choices=("core", "general", "refined"), default="core")
    parser.add_argument("--endpoint", choices=("ic50", "ec50"), default="ic50")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "sweeps")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--augmented-ratios", nargs="+", type=float, default=AUGMENTED_RATIOS)
    args, extra = parser.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]
    if args.max_parallel < 1:
        parser.error("--max-parallel must be at least 1")
    if any(ratio < 0 for ratio in args.augmented_ratios):
        parser.error("all --augmented-ratios must be non-negative")

    script = Path(__file__).resolve().parent / "train_ic50.py"
    jobs = []
    for domain, seed, ratio in itertools.product(args.domains, args.seeds, args.augmented_ratios):
        output_dir = args.output_root / args.endpoint / domain / f"augmented_ratio_{value_name(ratio)}" / f"seed_{seed}"
        command = [
            sys.executable, str(script), "--domain", domain, "--subset", args.subset, "--endpoint", args.endpoint,
            "--seed", str(seed), "--augmented-ratio", str(ratio),
            "--device", args.device, "--output-dir", str(output_dir),
        ]
        if args.data_root is not None:
            command.extend(("--data-root", str(args.data_root)))
        command.extend(extra)
        jobs.append((command, output_dir, domain, ratio))

    print(f"method=G-Mixup jobs={len(jobs)} max_parallel={args.max_parallel}")
    for command, *_ in jobs:
        print(" ".join(command))
    if args.dry_run:
        return

    def launch(job):
        return job, subprocess.run(job[0], check=False).returncode

    failures = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = [executor.submit(launch, job) for job in jobs]
        for future in as_completed(futures):
            job, returncode = future.result()
            if returncode:
                failures.append((job[0], returncode))
    if failures:
        details = "\n".join(f"exit={code}: {' '.join(command)}" for command, code in failures)
        raise SystemExit(f"{len(failures)}/{len(jobs)} jobs failed:\n{details}")

    grouped = {}
    for _, output_dir, domain, ratio in jobs:
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        grouped.setdefault((domain, ratio), []).append(summary)
    aggregate = {"method": "G-Mixup", "endpoint": args.endpoint, "seeds": args.seeds, "groups": {}}
    best = {}
    for (domain, ratio), summaries in sorted(grouped.items()):
        key = f"{domain}/augmented_ratio={ratio:g}"
        entry = {
            "runs": len(summaries),
            "ood_val_selection": stats(item["best_ood_val"] for item in summaries),
            "ood_test_accuracy": stats(item["metrics"]["ood_test"]["accuracy"] for item in summaries),
            "ood_test_roc_auc": stats(item["metrics"]["ood_test"]["roc_auc"] for item in summaries),
        }
        aggregate["groups"][key] = entry
        mean = entry["ood_val_selection"]["mean"]
        if mean is not None and (domain not in best or mean > best[domain][0]):
            best[domain] = (mean, key)
    aggregate["best_by_domain"] = {domain: key for domain, (_, key) in sorted(best.items())}
    path = args.output_root / args.endpoint / "aggregate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"aggregate={path}")


if __name__ == "__main__":
    main()
