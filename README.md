# Pneumonia Prediction

## Overview
This project utilises deep learning techniques to predict pneumonia from chest X-ray images. It includes data preprocessing, model training, evaluation, and visualization steps, leveraging convolutional neural networks (CNNs) for image classification.

## Dataset
The dataset used for this project is the **Chest X-Ray Pneumonia Dataset**, which contains labeled X-ray images of patients diagnosed with pneumonia and normal cases.

- **Source**: [Kaggle Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- The dataset is organized into training, testing, and validation subsets.

## Features
1. **Data Preprocessing**:
   - Images are resized to ensure uniformity.
   - Data augmentation techniques like flipping, rotation, and scaling are applied to enhance the dataset.

2. **Model Architecture**:
   - The model is built using a convolutional neural network (CNN).
   - Transfer learning with pre-trained models like VGG16 or ResNet is integrated to improve accuracy.

3. **Evaluation**:
   - Metrics like accuracy, precision, recall, and F1-score are used.
   - Confusion matrix and ROC curves are generated for analysis.

4. **Visualisation**:
   - Visualisation of training and validation loss/accuracy.
   - Examples of correctly and incorrectly classified images are displayed.

## Folder Structure
The following structure organizes the project files:

```
Pneumonia_Prediction/
├── Pneumonia_Prediction.ipynb  # Jupyter Notebook with the full project
├── models/                     # Trained model files
├── data/                       # Dataset folder (not included in repository)
│   ├── train/                  # Training data
│   ├── test/                   # Testing data
│   └── val/                    # Validation data
├── outputs/                    # Generated outputs (e.g., graphs, metrics)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Installation
To run this project, ensure you have Python 3.x installed along with the required libraries.

## Usage
1. Download the dataset from the [Kaggle link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and place it in the `data/` directory.
2. Open the Jupyter Notebook:

```bash
jupyter notebook Pneumonia_Prediction.ipynb
```

3. Run the cells in sequence to preprocess the data, train the model, and evaluate the results.

## Requirements
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn
- OpenCV

## Results
- The model achieves high accuracy in detecting pneumonia from X-ray images.
- Training and validation performance are visualized in the notebook.

## Acknowledgments
- Special thanks to [Paul Mooney](https://www.kaggle.com/paultimothymooney) for providing the dataset.
- Inspiration from academic studies and online tutorials on deep learning in medical imaging.

