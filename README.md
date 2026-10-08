# EGPL-AAC

Official research-code release for **Evidence-Grounded Preference Learning for Automated Audio Captioning**.

EGPL-AAC extends the EAT--Projector--Vicuna pipeline used by SLAM-AAC with:

1. an Event Residual Adapter (ERA), distilled from a frozen HTS-AT teacher; and
2. dataset-specific preference learning that updates only the existing LoRA parameters.

The final paper configuration uses ordinary DPO on AudioCaps and ordinary DPO with a lightweight one-sided chosen-caption anchor on Clotho. Evidence weights explored during ablation are not part of the final configuration.

## Release status

This repository contains the method-specific implementation, final hyperparameter records, preference-data schema, LoRA-only training utilities, and reported metric files. It is designed as an overlay for the upstream [SLAM-LLM/SLAM-AAC](https://github.com/X-LANCE/SLAM-LLM) codebase. Dataset manifests, pretrained models, generated preference data, and checkpoints are intentionally excluded.

The original experiment directory also contains exploratory BEATs, weighting, mention-selector, and graph branches. They are deliberately excluded because they are not part of the reported method. End-to-end launch scripts will be added only after machine-specific paths and upstream-code assumptions have been removed and clean-room reproduction has been verified.

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
  data.py                # AudioCaps/Clotho preference JSONL normalization
  preference.py          # length-normalized DPO and optional anchor
  training.py            # causal log-probabilities and LoRA-only freezing
configs/
  audiocaps_final.yaml
  clotho_final.yaml
results/
  final_metrics.json
  analysis_metrics.json
examples/
  preference_pair.example.jsonl
scripts/
  inspect_preference_data.py
tests/
  test_core.py
```

## Minimal validation

```bash
python -m pip install -e .
python -m pytest -q
python scripts/inspect_preference_data.py /private/path/to/preferences.jsonl
```

The preference loader accepts both final schemas. Required fields are
`key`, `source`, `chosen`, and `rejected`; evidence and filtering fields are
retained as audit metadata. The released example is synthetic and is not an
item from either training set.

The Clotho anchor exactly matches the final implementation: a linear,
one-sided hinge on the decrease in chosen-caption mean token log-likelihood,
with weight `0.01` and tolerance `0.02`. AudioCaps sets its weight to zero.

## Data and checkpoints

The following are not stored in Git:

- AudioCaps, Clotho, and preference JSONL files;
- EAT, HTS-AT, Vicuna, LoRA, and ERA checkpoints;
- decoded captions, caches, and experiment directories.

Do not commit licensed datasets or model weights. Configure their local paths only in private launch files or environment variables.

## Acknowledgement and license

This work builds on SLAM-LLM/SLAM-AAC. The included source code is released under the MIT License; see [LICENSE](LICENSE). Please also follow the licenses of SLAM-LLM, EAT, HTS-AT, Vicuna, and the datasets used in your environment.
