# CLASSIFYING THE SEVERITY OF DIABETIC RETINOPATHY FROM RETINAL FUNDUS IMAGES 

## NON-TECHNICAL EXPLANATION OF YOUR PROJECT
This project develops an AI model to automatically detect and classify diabetic retinopathy, a serious eye condition caused by diabetes, using retinal images. The model analyses small, coloured images of the back of the eye to identify five levels of severity: no disease, mild, moderate, severe, or proliferative. By training on a dataset of 2,400 images, the AI learns to spot patterns indicating disease severity, aiming to assist doctors in early diagnosis. Despite challenges with imbalanced data, the model achieves moderate accuracy, offering a foundation for future improvements in medical image analysis.

## DATA
The project uses the RetinaMNIST dataset, a subset of the MedMNIST v2 benchmark, containing 2,400 retinal fundus images (1,600 training, 400 validation, 400 test) resized to 64x64 pixels in RGB format. Each image is labelled with one of five diabetic retinopathy severity levels: No DR, Mild, Moderate, Severe, or Proliferative DR. The dataset is derived from the EyePACS Kaggle Diabetic Retinopathy Detection dataset, pre-processed for research use.

**Citation**:
- Yang, J., et al. "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification." Scientific Data 8, 207 (2021).
- EyePACS dataset: Kaggle Diabetic Retinopathy Detection (https://www.kaggle.com/c/diabetic-retinopathy-detection).

## MODEL
The model, `RetinaCNN_BO`, is a convolutional neural network (CNN) designed for lightweight classification of diabetic retinopathy on RetinaMNIST. It has three convolutional layers (with 16, 32, 64 filters in the best configuration), each followed by batch normalization, ReLU activation, and max-pooling, and two fully connected layers (128 units, then 5 units) with dropout (0.589). Focal loss and weighted sampling address class imbalance. This architecture was chosen for its simplicity, suitability for small datasets, and computational efficiency, making it ideal for rapid prototyping and educational purposes. 

## HYPERPARAMETER OPTIMISATION
The model optimizes five hyperparameters using Bayesian optimization with TPESampler over 50 trials:
- Learning rate (lr): [1e-5, 3e-4], best: 1.234e-05
- Batch size: [16, 32], best: 16
- Dropout rate: [0.3, 0.7], best: 0.589
- Number of filters: [[8, 16, 32], [16, 32, 64], [32, 64, 128]], best: [16, 32, 64]
- Weight decay: [0.0, 1e-3], best: 0.000565
Bayesian optimization was chosen to efficiently explore the hyperparameter space, maximizing validation accuracy while balancing computational cost. Focal loss parameters (gamma=3.0, alpha=[0.5, 1.0, 1.0, 2.0, 2.0]) were fixed to address class imbalance. 

## RESULTS
The model achieved a best validation accuracy of 56.67% (Trial 11) but a final validation accuracy of 50.00% and test accuracy of 45.00% on the 400-image test set, indicating retraining instability. The test confusion matrix shows strong performance on No DR (126/174 correct, ~72%) but poor results on minority classes (Mild: 16/46, ~34.8%; Moderate: 13/92, ~13.8%; Severe: 18/68, ~26.5%; Proliferative DR: 7/20, ~35.0%). This suggests the model struggles with class imbalance, despite focal loss and weighted sampling. The results highlight the need for stronger imbalance handling (e.g., higher gamma, adjusted alpha) and more epochs to stabilize training. 

![ConfustionMatrix](confusion_matrix.png)
