import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Water Potability App", page_icon="💧", layout="wide")

st.title("💧 Water Potability Prediction")
st.markdown("Predict whether water is **safe for drinking** using ML + WHO guidelines.")

# ===============================
# LOAD MODEL & SCALER
# ===============================
@st.cache_resource
def load_model():
    model = joblib.load("gb_model.pkl")
    scaler = joblib.load("scaler_water.pkl")
    return model, scaler

model, scaler = load_model()

def load_explainer(model):
    return shap.Explainer(model)
explainer=load_explainer(model)

# ===============================
# SIDEBAR INPUTS (ALL 9 FEATURES)
# ===============================
st.sidebar.header("🔍 Enter Water Parameters")

ph = st.sidebar.number_input("pH (6.5 – 8.5)", 0.0, 14.0, 7.0)
hardness = st.sidebar.number_input("Hardness (mg/L)", 0.0, 752.0, 150.0)
tds = st.sidebar.number_input("Total Dissolved Solids (mg/L)", 0.0, 1000.0, 300.0)
chlorine = st.sidebar.number_input("Chlorine (mg/L)", 0.0, 7.0, 0.5)
sulfate = st.sidebar.number_input("Sulfate (mg/L)", 0.0, 700.0, 100.0)

conductivity = st.sidebar.number_input("Conductivity (µS/cm)", 0.0, 4300.0, 300.0)
organic_carbon = st.sidebar.number_input("Organic Carbon (mg/L)", 0.0, 20.0, 10.0)
trihalomethanes = st.sidebar.number_input("Trihalomethanes (µg/L)", 0.0, 220.0, 50.0)
turbidity = st.sidebar.number_input("Turbidity (NTU)", 0.0, 10.0, 3.0)

predict_btn = st.sidebar.button("🚀 Predict")

# ===============================
# CREATE INPUT DATAFRAME
# ===============================
input_df = pd.DataFrame({
    "ph": [ph],
    "hardness": [hardness],
    "tds": [tds],
    "chlorine": [chlorine],
    "sulfate": [sulfate],
    "conductivity": [conductivity],
    "organic_carbon": [organic_carbon],
    "trihalomethanes": [trihalomethanes],
    "turbidity": [turbidity]
})

# ===============================
# MAIN PREDICTION SECTION
# ===============================
if predict_btn:
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled)[0][prediction]

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.success("✅ Water is Safe for Drinking")
        else:
            st.error("❌ Water is NOT Safe for Drinking")

    with col2:
        st.metric("Water Safety Probability", f"{confidence*100:.2f}%")

    st.subheader("📋 Entered Parameters")
    st.dataframe(input_df, use_container_width=True)

# ===============================
# TABS SECTION
# ===============================
tab1, tab2, tab3 = st.tabs(["📊 Feature Importance", "🧠 Model Info", "SHAP Explanation"])

# ===============================
# TAB 1: FEATURE IMPORTANCE
# ===============================
with tab1:
    st.subheader("Feature Importance")

    feature_names = [
        "ph", "hardness", "tds", "chlorine", "sulfate",
        "conductivity", "organic_carbon", "trihalomethanes", "turbidity"
    ]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots()
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")

    st.pyplot(fig)

# ===============================
# TAB 2: MODEL INFO
# ===============================
with tab2:
    st.subheader("Model Information")
    st.write("Model used: Gradient Boosting Classifier")
    st.write("This model predicts water potability based on chemical features.")

# ===============================
# TAB 3: WHO GUIDELINE CHECK
# ===============================
with tab3:
    st.subheader("SHAP Explanation")
    st.write("Shows how each features influenced this prediction.")

    if predict_btn:
        input_scaled=scaler.transform(input_df)

        shap_values=explainer(input_scaled)
        st.write("### Feature Contribution")

        shap_df=pd.DataFrame({
            "Feature": input_df.columns,
            "SHAP Value": shap_values.values[0]}).sort_values(by="SHAP Value",key=abs,ascending=False)

        def color_shap(val):
            if val > 0:
                return 'color: green'   # SAFE
            else:
                return 'color: red'     # NOT SAFE
        
        st.dataframe(
            shap_df.style.applymap(color_shap, subset=["SHAP Value"])
        )
        st.info(
            "Positive values push predictions towards 'Safe',"
            "Negative values push towards 'Not Safe'."
        )
    else:
        st.warning("Click 'predict' to see SHAP explanation")
    

   