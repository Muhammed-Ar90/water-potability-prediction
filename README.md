# 🚰 Water Potability Prediction using Machine Learning

A machine learning project that predicts whether water is safe for drinking based on physicochemical properties.  
This project covers an end-to-end pipeline including data preprocessing, model training, evaluation, and deployment for real-time predictions.

---

## 📸 App Preview

Below are sample outputs from the model interface:

### 🔹 Input Interface
![Input](assets/input.png)

### 🔹 Prediction Result
![Result](assets/result.png)

### 🔹 Model Insight
![Insight](assets/shap.png)

---

## 📌 Problem Statement

Safe drinking water is critical for public health.  
This project builds a classification model to determine:

- ✅ **1 → Safe for drinking**  
- ❌ **0 → Not safe for drinking**

based on measured water quality parameters.

---

## 📂 Dataset

The dataset used in this project is the **Water Potability Dataset**, commonly available on Kaggle.

It contains water quality measurements for different samples, along with a target variable indicating whether the water is safe for drinking.

### 📊 Features

The dataset includes the following 9 input features:

- pH  
- Hardness  
- Solids (Total Dissolved Solids - TDS)  
- Chloramines  
- Sulfate  
- Conductivity  
- Organic Carbon  
- Trihalomethanes  
- Turbidity  

### 🔗 Source

Dataset: https://www.kaggle.com/datasets/developerghost/water-potability

---

## 🔄 ML Pipeline

1. Data preprocessing (handling missing values, feature scaling using StandardScaler)
2. Model training using Gradient Boosting Classifier
3. Hyperparameter tuning (to optimize model performance)
4. Evaluation using classification metrics (Accuracy, Precision, Recall, F1 Score)
5. Deployment using Streamlit for real-time inference

---

## 🧠 Model Development

- **Algorithm:** Gradient Boosting Classifier  
- **Task:** Binary Classification  
- **Framework:** scikit-learn  

### Why Gradient Boosting?
- Captures non-linear relationships  
- Strong performance on tabular data  
- Handles feature interactions effectively  

---

## 📊 Model Evaluation

The model is evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1 Score
  
---

## 📊 Model Performance

- Accuracy: 99%
- Precision: 0.97- (The accuracy of the model when it predicts water is potable).
- Recall: 1.0 -(The model's ability to correctly identify all truly potable water samples).
- F1 Score: 0.98- (The balanced score between Precision and Recall, critical for imbalanced data).

---

## 📂 Project Structure

waterpotabilityy.py   # Inference app
gb_model.pkl          # Trained model
scaler_water.pkl      # Scaler
requirements.txt      # Dependencies
README.md

---

## 📈 Prediction Output

For a given input, the model returns:

- Water safety classification (Safe / Not Safe)  
- Prediction probability (**confidence score**)  

---

## ⚙️ Reproducibility & Environment

### 🐍 Python Version

This project is tested with:

> **Python 3.11**

⚠️ Newer versions may cause compatibility issues.

---

### 📦 Installation

```bash
pip install -r requirements.txt
```

## ▶️ Run the Project (Demo App)

After installing the dependencies, run the following command:

```bash
streamlit run waterpotabilityy.py
```
---

## 🚀 App Features
- Water quality prediction
- Confidence score
- SHAP-based model explanation
- Feature importance visualization
