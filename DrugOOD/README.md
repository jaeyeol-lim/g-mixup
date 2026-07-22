# G-Mixup on DrugOOD IC50

This directory adapts class-graphon G-Mixup to the shared DrugOOD IC50 protocol. It expects PyG caches named `drugood_lbap_{subset}_ic50_{domain}_{split}.pt`.

Single-run example:

```bash
python /workspace/baseline/g-mixup/DrugOOD/train_ic50.py \
  --data-root /workspace/Graph-OOD-Lab/data/DrugOOD \
  --domain assay \
  --subset core \
  --seed 1 \
  --augmented-ratio 0.25 \
  --device cuda:0 \
  --output-dir /workspace/baseline/g-mixup/DrugOOD/outputs/assay_ratio_0p25_seed_1
```

The search script evaluates augmented ratios `{0.15, 0.25, 0.5}`. Use `--dry-run` to print jobs without launching them. `DRUGOOD_DATA_ROOT` is supported when `--data-root` is omitted.

The original implementation targets topology-only TU datasets. For attributed DrugOOD graphs, this adaptation mixes degree-aligned class node-feature templates and class-average edge features with the same graphon interpolation coefficient. Only the original training split is used to estimate these quantities.
