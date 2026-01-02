# Semantic Segmentation of Road Scenes

This project explores **semantic segmentation of road scenes** using deep learning models implemented in **PyTorch**. The goal is to perform **pixel-level classification** on road images, allowing the model to distinguish roads, vehicles, pedestrians, buildings, and other scene elements.

The project was developed as part of the **Introduction to Neural Networks** course and focuses on building the full segmentation pipeline from data loading and preprocessing to model training and evaluation.

**Running the Project on Kaggle**

To run this project on Kaggle, the first step is to download the cityscapes-semantic-dataset from the Kaggle Datasets section. After adding the dataset to your Kaggle notebook, make sure it is available in the notebook’s input directory. Once the dataset is attached, you can run the notebook cells in order without modification, as the data loading and preprocessing steps are configured to work with the downloaded dataset structure.

---

## Project Overview

Semantic segmentation differs from image classification in that it assigns a class label to **every pixel** rather than a single label per image. This makes it especially important for applications such as autonomous driving and intelligent transportation systems, where understanding the structure of a scene is critical.

In this project, two convolutional neural network–based segmentation models were trained and compared:

* **FCN-ResNet50** (baseline model)
* **DeepLabV3-ResNet50** (advanced architecture with atrous convolutions)

Both models were trained and evaluated using the same data preprocessing pipeline and loss function to ensure a fair comparison.

---

## Dataset

The project uses the **Cityscapes / road-scene semantic segmentation dataset**, which contains:

* High-resolution RGB images of urban road scenes
* Pixel-wise semantic annotation masks
* Multiple semantic classes such as road, sidewalk, building, vehicle, pedestrian, and sky

Original Cityscapes label IDs were **remapped to 19 training classes**, with unused or void labels ignored during training using an `ignore_index` strategy.

---

## Methodology

### Data Preprocessing

* Images are resized to a fixed resolution and normalized using ImageNet statistics.
* Segmentation masks are resized using nearest-neighbor interpolation to preserve class labels.
* Masks are stored as integer class IDs (no normalization).

### Model Training

* Framework: **PyTorch**, **torchvision**
* Loss function: **Cross-Entropy Loss** with ignored void labels
* Optimizer: **Adam**
* Training performed on GPU when available (Kaggle environment)

### Evaluation

* Training loss is monitored across epochs.
* Model predictions are analyzed using **confusion matrices**.
* Pixel-level classification performance is assessed visually and quantitatively.

---

## Results

Both models successfully learned meaningful spatial representations of road scenes:

* **FCN-ResNet50** converged faster and provided a strong baseline.
* **DeepLabV3-ResNet50** achieved improved segmentation quality in complex regions due to its multi-scale context modeling.

Performance decreases in visually challenging areas such as shadows, occlusions, and object boundaries, which is expected in real-world road imagery.

---

## Key Takeaways

* Semantic segmentation requires careful handling of data preprocessing, especially for masks.
* Encoder-based CNN architectures can effectively perform dense pixel-level prediction.
* Architectural choices (e.g., atrous convolutions in DeepLabV3) have a noticeable impact on segmentation quality.
* Proper device management (CPU vs GPU) is critical for stable training and evaluation.

---

## Future Improvements

Possible extensions to this project include:

* Training for more epochs with learning-rate scheduling
* Adding data augmentation to improve generalization
* Computing additional metrics such as mean IoU across the full validation set
* Testing alternative architectures such as U-Net or attention-based segmentation models

---

## Course Information

* **Course:** Introduction to Neural Networks
* **Project Type:** Group Project (2 members)

