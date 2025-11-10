# 💰 Insurance Premium Predictor

A **Streamlit web app** that predicts individual insurance premium costs based on demographic and health-related features such as age, BMI, number of children, smoking status, and region.  
This project uses the **Kaggle Life Insurance dataset** to train regression models and provide interactive visualizations and predictions.

---

## 🧠 Overview

This app demonstrates a **machine learning regression workflow** built with **Python**, featuring:
- **Data loading & preprocessing**
- **Exploratory data analysis (EDA)**
- **Model selection and training**
- **Performance evaluation**
- **Feature importance visualization**
- **Interactive user prediction interface**

Users can choose between:
- **Linear Regression**
- **Random Forest Regressor**

to compare performance and understand feature impacts.

---

## 📊 Dataset

- **Source:** [Kaggle – Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **File:** `insurance.csv`

### **Features**

| Column | Description |
|--------|-------------|
| `age` | Age of the insured person |
| `sex` | Gender (`male`, `female`) |
| `bmi` | Body Mass Index (weight/height²) |
| `children` | Number of dependents covered by insurance |
| `smoker` | Smoking status (`yes`, `no`) |
| `region` | Residential area in the US |
| `charges` | Individual medical costs billed by health insurance |

---

## 🧩 Key Features

✅ **Interactive Visualizations**  
- Scatter plots for *BMI vs Charges* and *Age vs Charges*  
- Correlation heatmap for feature relationships  

✅ **Model Training & Evaluation**  
- Switch between Linear Regression and Random Forest  
- Adjustable hyperparameters (number of trees, max depth)  
- RMSE (Root Mean Squared Error) displayed for performance  

✅ **Feature Importance (Random Forest)**  
- Bar chart showing which features most influence predictions  

✅ **User Input Prediction Panel**  
- Input personal information (age, BMI, smoking, region, etc.)  
- Instant premium prediction with trained model  

---

## 🏗️ Installation & Setup

Follow these steps to set up and run the project locally.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/insurance-premium-predictor.git
cd insurance-premium-predictor
