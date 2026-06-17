# Child Marriage Risk Dashboard - Kenya
# Data Science Project - Phase 4 Deployment

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px

# Set up the page
st.set_page_config(
    page_title="Child Marriage Risk Predictor",
    page_icon="👧",
    layout="wide"
)

# Load model and files
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('child_marriage_model.pkl')
        scaler = joblib.load('scaler.pkl')
        with open('feature_names.json', 'r') as f:
            features = json.load(f)
        with open('model_metrics.json', 'r') as f:
            metrics = json.load(f)
        return model, scaler, features, metrics
    except FileNotFoundError:
        st.error("""
        Model files not found. Please make sure these files are in the same folder:
        - child_marriage_model.pkl
        - scaler.pkl
        - feature_names.json
        - model_metrics.json
        """)
        st.stop()

model, scaler, feature_names, metrics = load_artifacts()

# Title
st.title("Child Marriage Risk Assessment Tool")
st.markdown("""
This tool predicts child marriage rates for communities in Kenya based on key risk factors.
It is designed for NGOs, policymakers, and community leaders to identify high-risk areas.

**Important:** This tool predicts risk at the community level only. It does not identify individual girls at risk.
""")

# Sidebar inputs
st.sidebar.header("Community Profile")
st.sidebar.markdown("Enter your community characteristics:")

st.sidebar.subheader("Education")
no_education = st.sidebar.slider(
    "No Education Rate (%)",
    min_value=0, max_value=100, value=20, step=5
)

primary_education = st.sidebar.slider(
    "Primary Education Only (%)",
    min_value=0, max_value=100, value=50, step=5
)

st.sidebar.subheader("Location")
rural_pct = st.sidebar.slider(
    "Rural Population (%)",
    min_value=0, max_value=100, value=65, step=5
)

poorest_pct = st.sidebar.slider(
    "Poorest Population (%)",
    min_value=0, max_value=100, value=25, step=5
)

st.sidebar.subheader("Health")
adolescent_fertility = st.sidebar.number_input(
    "Adolescent Fertility Rate (per 1000)",
    min_value=0, max_value=200, value=85, step=5
)

st.sidebar.subheader("Empowerment")
literacy_rate = st.sidebar.slider(
    "Female Literacy Rate (%)",
    min_value=0, max_value=100, value=70, step=5
)

female_headed = st.sidebar.slider(
    "Female-Headed Households (%)",
    min_value=0, max_value=100, value=32, step=5
)

# Calculate derived feature
education_gap = (100 - no_education - primary_education) - no_education

# Create input
input_data = pd.DataFrame([[
    no_education,
    primary_education,
    rural_pct,
    poorest_pct,
    adolescent_fertility,
    literacy_rate,
    female_headed,
    education_gap
]], columns=feature_names)

# Make prediction
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

# Determine risk level
if prediction < 15:
    risk_tier = "LOW RISK"
    risk_color = "green"
    recommendation = "Maintain current programs. Monitor annually."
    priority = "Routine Monitoring"
elif prediction < 30:
    risk_tier = "MEDIUM RISK"
    risk_color = "orange"
    recommendation = "Targeted interventions needed in high-risk areas."
    priority = "Priority Intervention"
elif prediction < 50:
    risk_tier = "HIGH RISK"
    risk_color = "red"
    recommendation = "Immediate intervention needed. Prioritize resources."
    priority = "Urgent Priority"
else:
    risk_tier = "VERY HIGH RISK"
    risk_color = "darkred"
    recommendation = "Emergency intervention required."
    priority = "Emergency Priority"

# Display results
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Predicted Child Marriage Rate", f"{prediction:.1f}%")
with col2:
    st.metric("Risk Tier", risk_tier)
with col3:
    st.metric("Priority Level", priority)

# Risk gauge
st.markdown("---")
st.subheader("Risk Score Visualization")

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prediction,
    title={"text": "Risk Score (%)"},
    domain={'x': [0, 1], 'y': [0, 1]},
    gauge={
        'axis': {'range': [0, 80]},
        'bar': {'color': "coral"},
        'steps': [
            {'range': [0, 15], 'color': "#6bcb77"},
            {'range': [15, 30], 'color': "#ffd93d"},
            {'range': [30, 50], 'color': "#ff9f4a"},
            {'range': [50, 80], 'color': "#ff6b6b"}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 23
        }
    }
))

fig.update_layout(height=300)
st.plotly_chart(fig, use_container_width=True)

# Recommendations
st.markdown("---")
st.subheader("Recommended Actions")

if risk_color == "green":
    st.success(recommendation)
elif risk_color == "orange":
    st.warning(recommendation)
else:
    st.error(recommendation)

# Specific interventions
st.subheader("Suggested Interventions")

if no_education > 30:
    st.markdown("**Education Interventions:**")
    st.markdown("- Establish community-based girls' scholarship programs")
    st.markdown("- Build secondary schools within 5km radius")
    st.markdown("- Implement conditional cash transfers for school attendance")

if poorest_pct > 40:
    st.markdown("**Economic Interventions:**")
    st.markdown("- Vocational training programs for adolescent girls")
    st.markdown("- Microfinance groups for female-headed households")
    st.markdown("- Livelihood support for at-risk families")

if adolescent_fertility > 100:
    st.markdown("**Health Interventions:**")
    st.markdown("- Youth-friendly reproductive health services")
    st.markdown("- Mentorship programs for adolescent girls")
    st.markdown("- Access to sanitary products")

st.markdown("**Awareness Interventions:**")
st.markdown("- Community dialogues on child marriage laws")
st.markdown("- Engage religious and traditional leaders")
st.markdown("- Radio campaigns on girls' education rights")

# Feature importance
st.markdown("---")
st.subheader("What Drives Child Marriage?")

fi_data = metrics.get('feature_importance', [
    {'feature': 'No Education', 'importance': 0.31},
    {'feature': 'Rural Population', 'importance': 0.22},
    {'feature': 'Poverty', 'importance': 0.18}
])

fi_df = pd.DataFrame(fi_data)
fig2 = px.bar(fi_df, x='importance', y='feature', orientation='h',
              title="Top Risk Factors",
              color='importance', color_continuous_scale='Reds')
fig2.update_layout(height=350)
st.plotly_chart(fig2, use_container_width=True)

# Disclaimer
st.markdown("---")
st.caption(f"""
Model Performance: Mean Absolute Error is {metrics.get('test_mae', 6.8):.1f} percentage points.
Predictions may have error margins of approximately plus or minus {metrics.get('test_mae', 6.8):.1f} percent.

Need help? Contact your local children's office or police station.

For planning and resource allocation only. Not for individual risk assessment.
""")