# 🚀 ANN Churn Prediction

## 🧠 Overview
Customer churn prediction is crucial for companies that depend on recurring customers, such as telecom or subscription-based services.  
This project implements an **Artificial Neural Network (ANN)** using **TensorFlow/Keras** to predict whether a customer will leave (churn) or stay.  

The notebook `ANN_Churn_Prediction.ipynb` contains a complete, step-by-step deep learning workflow — from **data preprocessing** to **model evaluation** and **performance visualization**.

---

## 🌟 Key Features
- ✅ End-to-end churn prediction pipeline  
- 🧩 Clean TensorFlow/Keras implementation  
- 📊 Detailed performance metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix  
- 🔍 Preprocessing steps with one-hot encoding and feature normalization  
- 🧱 Modular design ready for experimentation and hyperparameter tuning  

---

## 📂 Table of Contents
- [About](#-about)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Installation](#️-installation)
- [Usage](#️-usage)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#️-license)
- [Contact](#-contact)

---

## ❓ About
The goal of this project is to train a deep learning model capable of **predicting customer churn** based on customer attributes such as demographics, subscription type, and service usage.  
It is designed for **educational and research purposes**, demonstrating how to build and interpret ANN models for classification problems.

**Performance Summary:**

| Metric | Value |
|---------|--------|
| Accuracy | ~79% |
| Precision | 0.65 – 0.83 |
| Recall | 0.58 – 0.87 |
| F1 Score | 0.73 – 0.85 |

---

## 📊 Dataset
- **Dataset Name:** Telco Customer Churn Dataset  
- **Source:** [`WA_Fn-UseC_-Telco-Customer-Churn.csv`](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- **Features Include:**
  - Customer demographics (gender, age, location)
  - Subscription and service details (internet, streaming, phone)
  - Payment and contract types
  - Churn label (Yes/No)

**Data Preprocessing Steps:**
- Removal of missing or invalid entries  
- One-hot encoding of categorical variables  
- Feature scaling using normalization  
- Train-test split (80/20)

---

## 🧭 Project Workflow
1. **Data Loading & Exploration**
   - Import dataset, visualize distributions, and handle missing values.  
2. **Preprocessing**
   - Encode categorical features and normalize numerical ones.  
3. **Model Building**
   - Design an ANN using Keras with multiple dense layers.  
4. **Training & Evaluation**
   - Train using the Adam optimizer and binary cross-entropy loss.  
   - Evaluate using confusion matrix, classification report, and accuracy plots.  
5. **Visualization**
   - Graphs for training accuracy, loss, and key metrics.

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ann-churn-prediction.git
   cd ann-churn-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add dataset**
   Download the dataset file (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) and place it in the project root directory.

---

## ▶️ Usage

1. Open the notebook:
   ```bash
   jupyter notebook ANN_Churn_Prediction.ipynb
   ```
2. Run all cells to preprocess data, train the model, and evaluate performance.  
3. View accuracy, precision, recall, F1-score, and confusion matrix in the output cells.  
4. You can adjust hyperparameters (epochs, layers, neurons) in the notebook for experimentation.

---

## 🏗️ Model Architecture

| Layer | Units | Activation | Description |
|--------|--------|-------------|--------------|
| Input Layer | 26 | - | 26 input features |
| Hidden Layer 1 | 26 | ReLU | Dense layer |
| Hidden Layer 2 | 15 | ReLU | Dense layer |
| Output Layer | 1 | Sigmoid | Binary classification output |

**Optimizer:** Adam  
**Loss:** Binary Crossentropy  
**Epochs:** 100

---

## 📈 Results

| Metric | Value |
|---------|--------|
| Accuracy | ~79% |
| Precision | 0.65 – 0.83 |
| Recall | 0.58 – 0.87 |
| F1 Score | 0.73 – 0.85 |

The confusion matrix, training accuracy/loss curves, and classification report are displayed in the notebook.

## 🤝 Contributing
Contributions are welcome!  

1. Fork the repository  
2. Create a new branch (`git checkout -b feature/your-feature`)  
3. Commit your changes (`git commit -m 'Add feature'`)  
4. Push to the branch (`git push origin feature/your-feature`)  
5. Open a **Pull Request**

---
