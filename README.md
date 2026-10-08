# EGPL-AAC

Official research-code release for **Evidence-Grounded Preference Learning for Automated Audio Captioning**.

EGPL-AAC extends the EAT--Projector--Vicuna pipeline used by SLAM-AAC with:

1. an Event Residual Adapter (ERA), distilled from a frozen HTS-AT teacher; and
2. dataset-specific preference learning that updates only the existing LoRA parameters.

The final paper configuration uses ordinary DPO on AudioCaps and ordinary DPO with a lightweight one-sided chosen-caption anchor on Clotho. Evidence weights explored during ablation are not part of the final configuration.

## Release status

This repository currently contains the method-specific implementation, final hyperparameter records, and reported metric files. It is designed as an overlay for the upstream [SLAM-LLM/SLAM-AAC](https://github.com/X-LANCE/SLAM-LLM) codebase. Dataset manifests, pretrained models, generated preference data, and checkpoints are intentionally excluded.

The end-to-end launch scripts and data-preparation instructions will be added after all machine-specific paths have been removed and the clean-room reproduction commands have been verified.

## Final configurations

| Dataset | Residual scale | Preference objective | Pairs | Beta | Learning rate | Updates | Seed |
|---|---:|---|---:|---:|---:|---:|---:|
| AudioCaps | 0.1 | ordinary DPO | 512 | 0.05 | 1e-6 | 64 | 43 |
| Clotho | 0.1 | DPO + chosen-caption anchor (0.01) | 4,345 | 0.05 | 2e-6 | 544 | 42 |

Both use an ERA with hidden size 256, kernel size 5, three temporal blocks, 527 AudioSet event classes, and 64 temporal bins.

## Repository layout

```text
egpl_aac/
  event_adapter.py       # ERA architecture and temporal event residual
  preference.py          # length-normalized DPO and optional anchor
configs/
  audiocaps_final.yaml
  clotho_final.yaml
results/
  final_metrics.json
tests/
  test_core.py
```

## Minimal validation

```bash
python -m pytest -q
```

## Data and checkpoints

The following are not stored in Git:

- AudioCaps, Clotho, and preference JSONL files;
- EAT, HTS-AT, Vicuna, LoRA, and ERA checkpoints;
- decoded captions, caches, and experiment directories.

Do not commit licensed datasets or model weights. Configure their local paths only in private launch files or environment variables.

## Acknowledgement and license

This work builds on SLAM-LLM/SLAM-AAC. The included source code is released under the MIT License; see [LICENSE](LICENSE). Please also follow the licenses of SLAM-LLM, EAT, HTS-AT, Vicuna, and the datasets used in your environment.
