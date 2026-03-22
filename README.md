# DeepLense GSoC 2026 - Evaluation Tests

**Author:** Prakhar Prakarsh
**Email:** prakharprakarsh@gmail.com
**Organisation:** Machine Learning for Science (ML4SCI)
**Projects:** DEEPLENSE6 (Gravitational Lens Finding) and DEEPLENSE4 (Foundation Model)

## Test Results

### Common Test I: Multi-Class Classification
- **Model:** ResNet-18 (ImageNet pretrained, fine-tuned)
- **Accuracy:** 92.64%
- **Macro-Average AUC:** 0.9869

### Specific Test V: Lens Finding
- **Model:** ResNet-18 with weighted CrossEntropyLoss
- **AUC Score:** 0.9868
- **Recall at FPR=0.01:** 84.6%
- **Best F1-Score:** 0.7324 (optimized threshold)

## Repository Contents
- `Common_Test_I_MultiClass_Classification.ipynb` - 3-class classification of lensing images
- `Test_V_Lens_Finding.ipynb` - Binary lens finding with extreme class imbalance
- `resnet18_deeplense_classifier.pth` - Trained model weights (Test I)
- `resnet18_lens_finder.pth` - Trained model weights (Test V)
