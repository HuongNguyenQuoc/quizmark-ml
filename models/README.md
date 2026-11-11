# Bubble classifier models

`bubble_classifier_v1.onnx` and `model_card.json` are the practice export from
Colab on 2026-11-11, using training code at commit `fae7b0b`.

- Trained only on generated sheet images: 74 sheets / 28,260 cells.
- Train / validation / test: 20,376 / 3,310 / 4,574 cells, split by sheet.
- Classes: `empty`, `filled`, `erased`; input `[1, 1, 32, 32]`, grayscale,
  `INTER_AREA` resize, float32 / 255. ONNX opset 11.
- Synthetic test accuracy: 99.803%; filled/not-filled errors: model 9, old rule 150.
- Model SHA-256: `e675d8b32826f8964bf48d129fecf4831f132d60c6889b6b152b685ce8e0b563`.

This model supports development and phone comparison tests. Real paper scans
are still required before enabling it in a release build. The app enables it
by default only in debug builds and provides an Account switch for comparison.

`verification_v1.json` records local OpenCV 4.5.3 compatibility checks and a
repeat evaluation of the held-out data, including the app's 0.6-confidence fallback.
The fallback policy also makes 9 filled/not-filled errors on this test set.
The random sheet split has no template-4 (50-question) sheet in the final test
set; validation contains that template. Collect a broader real-data test set
before interpreting the aggregate accuracy as coverage of all templates.
