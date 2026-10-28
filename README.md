---
title: PY18 Research Console
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# PY18 Research Console

A Gradio-based research prototype for experimental image classification and model evaluation.

## Features
- DenseNet-169 image prediction using a checkpoint from Hugging Face Hub
- Qwen3 1.7B research assistant, loaded only when first used
- Preliminary model comparison
- Prediction report exports
- Model registry / knowledge store

## Model repositories
- DenseNet-169: Mpvr/autism-densenet169
- Qwen3 1.7B: Mpvr/autism-qwen3-1.7b

## Notes
Predictions are experimental and must not be used for medical diagnosis. Comparison figures are preliminary and should be verified against actual evaluation results.

The models are loaded on demand to reduce startup memory usage. Free CPU hardware may still be insufficient for the language model, depending on model weights and available memory.
