# TrashNet Model Training Repository

This repository contains the scripts and notebooks to train a classification model for the TrashNet dataset using PyTorch and track performance using Weights & Biases (WandB).

## Table of Contents

1. [Setup](#setup)
2. [Dataset](#dataset)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [How to Reproduce](#how-to-reproduce)

---

## Setup

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- Pip
- GPU-enabled machine (optional but recommended for faster training)

### Installation

Clone the repository and install the required dependencies:

```bash
# Clone this repository
git clone https://github.com/your-username/trashnet-classifier.git
cd trashnet-classifier

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

All required dependencies are listed in `requirements.txt`.

```txt
torch
torchvision
torchmetrics
scikit-learn
matplotlib
pandas
wandb
```

---

## Dataset

The TrashNet dataset is used for this project. The dataset includes images of trash categorized into multiple classes.

- The dataset is loaded using the Hugging Face `datasets` library.
- Images are preprocessed using PyTorch's `transforms` module to resize, normalize, and convert them to tensors.

---

## Training

The model used is a pre-trained ResNet18, fine-tuned for the TrashNet dataset. The key steps include:

1. **Data Preprocessing**:
   - Resize images to `224x224`.
   - Normalize pixel values.
2. **Data Splitting**:
   - Split the dataset into training (80%) and validation (20%) sets.
3. **Model Training**:
   - Train the model using the CrossEntropy loss function and Adam optimizer.
   - Log metrics (loss, validation accuracy) using WandB.

---

## Evaluation

The model's performance is evaluated based on:

- **Validation Accuracy**: Accuracy on the validation set.
- **Loss Curves**: Monitor loss curves to detect overfitting or underfitting.

---

## How to Reproduce

Follow these steps to reproduce the training process:

1. Clone this repository and navigate to the root directory:

   ```bash
   git clone https://github.com/your-username/trashnet-classifier.git
   cd trashnet-classifier
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook to train the model:

   ```bash
   jupyter notebook TrashNet_Training.ipynb
   ```

4. (Optional) Log into Weights & Biases:

   ```bash
   wandb login
   ```

5. Run the script:

   ```bash
   python train.py
   ```

---

## Repository Structure

```plaintext
.
├── TrashNet_Training.ipynb  # Jupyter notebook for training and evaluation
├── train.py                 # Python script for training
├── requirements.txt         # Required dependencies
├── README.md                # Documentation
```

---

## Results

- Final Validation Accuracy: ~90%
- Train Loss: Converged below 0.1
