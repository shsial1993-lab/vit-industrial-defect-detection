# Tiny ViT Industrial Defect Detection

A compact Vision Transformer baseline for image-level industrial defect classification.
The project is intentionally dataset-agnostic: the demo uses deterministic synthetic
images so the architecture can be tested without downloading proprietary inspection data.

## What it demonstrates

- Patch embedding and a learnable class token
- Transformer encoder blocks for visual features
- A reproducible training/inference entry point
- A clean place to add real inspection data, augmentations, and calibration

## Run

~~~bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m src.demo
~~~

This is a research/portfolio baseline, not a claim about production accuracy. Replace
the synthetic loader with a labeled inspection dataset before reporting results.

## Structure

- src/model.py — tiny ViT architecture
- src/demo.py — synthetic training smoke test
- tests/test_model.py — shape and gradient checks

## License

Apache-2.0
