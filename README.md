# ML4SCI DeepLense — Evaluation Test Submissions

**Author:** Prakhar Prakarsh · [prakharprakarsh@gmail.com](mailto:prakharprakarsh@gmail.com)
**Organisation:** Machine Learning for Science (ML4SCI)
**Context:** Evaluation tasks completed for the ML4SCI DeepLense GSoC 2026 application
(DEEPLENSE6 — Gravitational Lens Finding; DEEPLENSE4 — Foundation Model).

Gravitational-lensing image classification using deep learning (PyTorch, ResNet-18,
transfer learning). Two tasks: multi-class classification and binary lens finding
under extreme class imbalance.

## Results

### Common Test I — Multi-Class Classification
- **Model:** ResNet-18 (ImageNet-pretrained, fine-tuned)
- **Accuracy:** 92.64%
- **Macro-average AUC:** 0.9869

### Specific Test V — Lens Finding
- **Model:** ResNet-18 with weighted cross-entropy loss
- **AUC:** 0.9868
- **Recall @ FPR = 0.01:** 84.6%
- **Best F1 (optimised threshold):** 0.7324

## Repository Contents
- `README.md` — this file
- `Common_Test_I_MultiClass_Classification.ipynb` — 3-class classification of lensing images
- `Test_V_Lens_Finding.ipynb` — binary lens finding under extreme class imbalance

## Reproducing
Open either notebook in Jupyter or Google Colab and run top to bottom.
Dependencies: `torch`, `torchvision`, `numpy`, `scikit-learn`, `matplotlib`.
