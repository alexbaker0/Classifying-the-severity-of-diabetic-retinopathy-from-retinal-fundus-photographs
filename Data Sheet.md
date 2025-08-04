# Classifying severity of diabetic retinopathy from retinal fundus images - DATASHEET

## Motivation

### Purpose:

The RetinaMNIST dataset was created to enable the development and benchmarking of machine learning models—especially CNNs—for the automatic classification and grading of diabetic retinopathy (DR) in color retinal fundus images. The goal is to facilitate research in automated medical image analysis, improve clinical decision support, and provide a standardized benchmark for algorithm comparison. The dataset is designed to be lightweight (small image size, manageable dataset size) for rapid prototyping and educational purposes, as part of the MedMNIST v2 benchmark suite.

### Creators, Funding:

RetinaMNIST is a subset of the MedMNIST benchmark, curated by the MedMNIST team led by researchers from Zhejiang University and associated collaborators, notably Junhao Yang and colleagues.

* Source publication: Yang, J., et al. "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification." Scientific Data 8, 207 (2021).
* The original dataset ("Kaggle Diabetic Retinopathy Detection") was made available via Kaggle, contributed by EyePACS.
* Funding: Dataset and MedMNIST supported by academic/research funding sources (details may be found in the publications). Specific funding details are not publicly disclosed in the source publication, but likely include grants from Zhejiang University or collaborating institutions.

## Composition

### Instances Represent:

Each instance is a 64×64 pixel RGB image of a human retina (fundus photograph), labeled with the grade of diabetic retinopathy present.

### Categories:

5 classes representing DR severity:

* 0 - No DR
* 1 - Mild
* 2 - Moderate
* 3 - Severe
* 4 - Proliferative DR

### Instance Counts:

Training: 1,600 images
Validation: 400 images
Test: 400 images
The class distribution is approximately: No DR (~50%), Mild (~10%), Moderate (~20%), Severe (~10%), Proliferative DR (~10%). 

### Missing Data:

No known missing image data in the public version; all instances include an associated label.

### Confidentiality:

Images are de-identified prior to dataset release. No personal information or protected health information (PHI) is included. Images are considered low risk for privacy when used for research or educational purposes. The de-identification process by EyePACS ensures compliance with HIPAA and other privacy regulations, removing patient identifiers from metadata and images.

## Collection Process

### Source:

Images are a downsampled and pre-processed version of the EyePACS Kaggle diabetic retinopathy images, which are clinical photographs collected as part of diabetic retinopathy screening programs.

### Acquisition:

Images and labels were taken from the online MNIST repository and, specifically, the set of 1,600 64x64 colour images

### Sampling Strategy:

Original sampling and inclusion criteria are set by the EyePACS/Kaggle organizers; MedMNIST applied further curation for class balance and coverage. EyePACS collected images from clinical settings, likely with stratified sampling to ensure representation of all DR severity levels, though exact criteria are not fully documented. MedMNIST further subsampled to create a balanced, lightweight dataset, but severe class imbalance remains (e.g., No DR dominates). The standard test / train / validation set split from the RetinaMNSIT dataset was used.

### Time Frame:

EyePACS dataset: images collected pre-2015 for the original Kaggle DR Challenge. MedMNIST construction completed in 2021.

## Preprocessing/Cleaning/Labelling

### Preprocessing:

All images were:

* Cropped from full fundus photos to centred retinal region,
* Down sampled (original images much larger),
* Rescaled or resized to 28x28, 64x64 and 128x128
* Converted to RGB (3 channels).
* Values cast to float32 and normalized to \[-1,1]
  Labels correspond to the verified clinical diabetic retinopathy grade per image. Additional preprocessing includes enhanced data augmentation for training (RandomHorizontalFlip(p=0.5), RandomRotation(45), RandomResizedCrop(64, scale=(0.6, 1.0)), ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1), RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)), ToTensor()), and minimal preprocessing for validation/test (ToTensor()). 

### Saved Raw Data:

MedMNIST provides preprocessed images only. Original EyePACS data may be accessible via Kaggle (license required). The raw EyePACS images (high-resolution fundus photos) are available through Kaggle, but access requires agreement to their terms. Only the pre-processed impages from the RetinaMNIST dataset were used

## Uses

### Other Tasks:

* General benchmarking of classification methods for biomedical images.
* Research in explainable AI, transfer learning, robustness testing.
* Educational demonstrations and model prototyping for medical image analysis.
* The dataset can also be used for tasks like anomaly detection, feature extraction for other retinal diseases, or testing domain adaptation techniques.

### Potential Bias/Impact:

* As with most clinical imaging datasets, limited demographic information may induce bias (patient populations, scanner vendors, etc.). The EyePACS dataset primarily includes images from U.S.-based clinical settings, which may not represent global populations or diverse imaging equipment, potentially affecting model generalizability.
* Task is not for clinical deployment. Results may differ by population or imaging device not represented in dataset. The model’s training output shows severe class imbalance (e.g., No DR dominates, with poor performance on Mild, Moderate, Severe, Proliferative DR), which could lead to biased predictions favoring the majority class. Dataset consumers should be aware of this imbalance and use techniques like weighted sampling or focal loss to mitigate it.
* Mitigation: Dataset consumers can apply oversampling, weighted loss functions (e.g., FocalLoss with gamma=4.0, alpha=\[0.2, 1.0, 1.0, 4.0, 4.0]), or data augmentation to address imbalance. External validation on diverse datasets (e.g., other DR datasets like APTOS) can improve generalizability. No specific  specific bias mitigation strategies were applied.

### Not-for-Use:

* Not to be used for clinical or financial decision-making.
* Not suitable for any real-time patient care scenarios.
* Not for commercial redistribution outside MedMNIST license.
* Additionally, the dataset should not be used for tasks requiring high-resolution details (e.g., microaneurysm detection), as the 64x64 resolution limits fine-grained feature extraction.

## Distribution

### Distribution:

MedMNIST (incl. RetinaMNIST) is distributed at https://medmnist.com/, via Zenodo and GitHub. The dataset can be downloaded as a NumPy archive (.npz) or accessed programmatically via the MedMNIST Python package. I use the MedMNIST package or a custom loader to access the 64x64 images.

### Licensing/ToU:

* MedMNIST datasets are released under a Creative Commons Attribution 4.0 International License.
* Original EyePACS/Kaggle data governed by Kaggle Data License. The Kaggle license requires users to agree to terms for non-commercial use and proper attribution.

## Maintenance

Maintenance is by the MedMNIST team, led by Junhao Yang et al. Contact information available on the MedMNIST official site. The team provides updates via GitHub (https://github.com/MedMNIST/MedMNIST) and responds to issues or questions through the repository or email (contact@medmnist.com).

