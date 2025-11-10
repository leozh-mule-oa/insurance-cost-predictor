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

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt

If you don't have a requirement.txt file yet, you can create one with the following contents:

```bash
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn

### 3️⃣ Download the Dataset

Go to the official dataset page:
👉 Medical Cost Personal Dataset (Kaggle) (https://www.kaggle.com/datasets/mirichoi0218/insurance)

Click Download and extract the insurance.csv file.

Place insurance.csv in the same folder as your Streamlit script (e.g., insurance_app.py).

```bash
insurance-premium-predictor/
│
├── insurance_app.py
├── insurance.csv
├── requirements.txt
└── README.md

4️⃣ Run the Streamlit App

Run the app locally using the command below:

```bash
streamlit run insurance_app.py

THen open your browser and navigate to: 

```bash
http://localhost:8501

⚙️ How It Works

Data Loading: Loads insurance.csv into a Pandas DataFrame.

Preprocessing: Encodes categorical features (sex, smoker, region).

Visualization: Displays data relationships using Matplotlib and Seaborn.

Model Training:

Train/test split (80/20).

Fit the selected regression model.

Performance Metrics: Compute MSE and RMSE.

Prediction Interface: Collects user inputs and outputs an estimated insurance cost.

📉 Example Output

RMSE (Linear Regression): ~$6,000–$7,000

RMSE (Random Forest): ~$4,000–$5,000
(values may vary by random state and parameters)

❤️ Credits

Dataset: Kaggle – Insurance Cost Dataset by Mirichoi0218

Libraries: Streamlit, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

Developed by: Leo Zhang

📜 License

This project is open-sourced under the MIT License.
Feel free to fork, modify, and use it for learning or demonstrations.