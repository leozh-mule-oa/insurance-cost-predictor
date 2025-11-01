import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# -------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="💰 Insurance Cost Predictor",
    layout="wide",
    page_icon="💖"
)
st.title("💰 Insurance Cost Prediction App")
st.write("Predict individual insurance costs based on demographic & health-related features.")

# -------------------------------
# 2. LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("insurance.csv")  # Ensure insurance.csv is in the same folder https://www.kaggle.com/datasets/mirichoi0218/insurance?resource=download
    return df

df = load_data()
st.subheader("📊 Data Preview")
st.dataframe(df.head(1000), use_container_width=True)

# -------------------------------
# 3. DATA PREPROCESSING
# -------------------------------
df_processed = df.copy()
# Encode categorical variables
df_processed['sex'] = df_processed['sex'].map({'male': 0, 'female': 1})
df_processed['smoker'] = df_processed['smoker'].map({'no': 0, 'yes': 1})
df_processed = pd.get_dummies(df_processed, columns=['region'], drop_first=True)

feature_cols = ['age', 'sex', 'bmi', 'children', 'smoker',
                'region_northwest', 'region_southeast', 'region_southwest']
X = df_processed[feature_cols]
y = df_processed['charges']

# -------------------------------
# 4. VISUALIZATION
# -------------------------------
st.subheader("📈 Exploratory Visualizations")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**BMI vs Charges**")
    fig, ax = plt.subplots()
    ax.scatter(df['bmi'], df['charges'], alpha=0.6, color="#1e90ff")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Charges ($)")
    st.pyplot(fig, use_container_width=True)

with col2:
    st.markdown("**Age vs Charges**")
    fig, ax = plt.subplots()
    ax.scatter(df['age'], df['charges'], alpha=0.6, color="#ffa502")
    ax.set_xlabel("Age")
    ax.set_ylabel("Charges ($)")
    st.pyplot(fig, use_container_width=True)

st.markdown("**Correlation Heatmap**")
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(df_processed.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
            annot_kws={"size": 7}, cbar_kws={"shrink": 0.5})
fig.tight_layout(pad=0.5)
st.pyplot(fig, use_container_width=False)

# -------------------------------
# 5. MODEL SELECTION & TRAINING
# -------------------------------
st.markdown("""
<div style='background-color:#f0f8ff; padding:10px; border-radius:10px;'>
<h3 style='color:#1e90ff; text-align:center;'>⚙️ Select a Model to Train</h3>
</div>
""", unsafe_allow_html=True)

model_name = st.radio("", ("Linear Regression", "Random Forest Regressor"), index=0, horizontal=True)

if model_name == "Random Forest Regressor":
    n_estimators = st.slider("Number of Trees (n_estimators):", 50, 500, 100, 50)
    max_depth = st.slider("Max Depth:", 2, 20, 6)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if model_name == "Linear Regression":
    model = LinearRegression()
else:
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

model.fit(X_train, y_train)
preds = model.predict(X_test)

mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)

st.subheader("🧮 Model Performance")
#st.info(f"Mean Squared Error (MSE): **{mse:,.2f}**")
st.success(f"Root Mean Squared Error (RMSE): **${rmse:,.2f}**")

# -------------------------------
# 6. MODEL DIAGNOSTICS
# -------------------------------
st.subheader("📉 Model Diagnostics")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Predicted vs Actual Charges**")
    fig, ax = plt.subplots(figsize=(5,4))
    ax.scatter(y_test, preds, alpha=0.6, color="#2ed573")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    ax.set_xlabel("Actual Charges ($)")
    ax.set_ylabel("Predicted Charges ($)")
    ax.set_title(f"{model_name} Fit")
    st.pyplot(fig, use_container_width=True)

with col2:
    st.markdown("**Residuals Distribution**")
    residuals = y_test - preds
    fig, ax = plt.subplots(figsize=(5,4))
    sns.histplot(residuals, bins=20, kde=True, color="#1e90ff", ax=ax)
    ax.set_xlabel("Prediction Error (Actual - Predicted)")
    ax.set_ylabel("Count")
    ax.set_title("Residuals Histogram")
    st.pyplot(fig, use_container_width=True)

st.markdown("""
✅ **Interpretation:**
- Scatter points near the red line → good fit.
- Residuals centered around 0 → unbiased.
- Skewed residuals → model under/overestimates some cases.
""")

# -------------------------------
# 7. USER INPUT & PREDICTION
# -------------------------------
st.subheader("🎯 Predict Your Premium / Cost")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    age = st.slider("Age", 18, 100, 30)
with col2:
    bmi = st.slider("BMI", 10, 60, 27)
with col3:
    children = st.slider("Number of Children", 0, 10, 0)
with col4:
    smoker_option = st.selectbox("Smoker?", ("No", "Yes"))
    smoker_val = 1 if smoker_option == "Yes" else 0
with col5:
    sex_option = st.selectbox("Sex", ("Male", "Female"))
    sex_val = 1 if sex_option == "Female" else 0

region_option = st.selectbox("Region", ("southwest", "southeast", "northwest", "northeast"))
region_dict = {"southwest":[1,0,0], "southeast":[0,1,0], "northwest":[0,0,1], "northeast":[0,0,0]}
region_vals = region_dict[region_option]

input_dict = {
    'age': age,
    'sex': sex_val,
    'bmi': bmi,
    'children': children,
    'smoker': smoker_val,
    'region_northwest': region_vals[2],
    'region_southeast': region_vals[1],
    'region_southwest': region_vals[0]
}

# **FIX:** Ensure columns match training feature order
input_df = pd.DataFrame([input_dict])[feature_cols]

pred_input = model.predict(input_df)[0]
st.success(f"💰 Estimated Annual Cost: **${pred_input:,.2f}**")

# -------------------------------
# 8. FOOTER
# -------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Seaborn, and Scikit-learn with real insurance data.")
