
import streamlit as st
import pandas as pd
import joblib
import io
from datetime import datetime

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="HealthPredict | Diabetes Analytics",
    page_icon="🩺",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

model = joblib.load("diabetes_prediction_model.pkl")

# ---------------- SESSION HISTORY ----------------

if "history" not in st.session_state:
    st.session_state.history = []

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>

.stApp {
    background-color: #F3F6FA;
}

[data-testid="stSidebar"] {
    background-color: #101D35;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 8px;
}

.main-title {
    color: #142B4A;
    font-size: 34px;
    font-weight: 750;
    margin-bottom: 0;
}

.subtitle {
    color: #718096;
    font-size: 15px;
    margin-top: 4px;
}

.section-title {
    color: #142B4A;
    font-size: 22px;
    font-weight: 700;
}

.metric-card {
    background-color: #FFFFFF;
    border: 1px solid #E3EAF2;
    border-radius: 15px;
    padding: 22px;
    min-height: 115px;
}

.metric-label {
    color: #718096;
    font-size: 14px;
}

.metric-value {
    color: #142B4A;
    font-size: 28px;
    font-weight: 750;
    margin-top: 8px;
}

.info-card {
    background-color: #FFFFFF;
    border: 1px solid #E3EAF2;
    border-radius: 15px;
    padding: 22px;
}

.stButton > button {
    background-color: #087E8B;
    color: white;
    border: none;
    border-radius: 9px;
    height: 45px;
    font-weight: 650;
}

.stButton > button:hover {
    background-color: #06636D;
    color: white;
    border: none;
}

.stDownloadButton > button {
    background-color: #142B4A;
    color: white;
    border-radius: 9px;
}

div[data-testid="stForm"] {
    background-color: white;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #E3EAF2;
}

</style>
""", unsafe_allow_html=True)


# ---------------- HELPER FUNCTIONS ----------------

def metric_card(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def create_pdf(result):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=(8.27 * inch, 11.69 * inch)
    )

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph("HealthPredict", styles["Title"])
    )

    elements.append(
        Paragraph("Diabetes Prediction Report", styles["Heading2"])
    )

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph(
            "Generated on: " + result["Date"],
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 20))

    table_data = [
        ["Patient Information", "Entered Value"],
        ["Age", str(result["Age"])],
        ["Gender", result["Gender"]],
        ["BMI", str(result["BMI"])],
        ["Glucose", str(result["Glucose"])],
        ["Blood Pressure", str(result["Blood Pressure"])],
        ["Cholesterol", str(result["Cholesterol"])],
        ["Insulin", str(result["Insulin"])],
        ["HbA1c", str(result["HbA1c"])],
        ["Smoking Status", result["Smoking Status"]],
        ["Physical Activity", result["Physical Activity"]],
        ["Family History", result["Family History"]]
    ]

    table = Table(table_data, colWidths=[220, 220])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#142B4A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE3EC")),
        ("PADDING", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white)
    ]))

    elements.append(table)
    elements.append(Spacer(1, 25))

    elements.append(
        Paragraph("Prediction Result", styles["Heading2"])
    )

    elements.append(
        Paragraph(result["Prediction"], styles["Normal"])
    )

    if result["Probability"] is not None:
        elements.append(
            Paragraph(
                "Model-estimated probability: "
                + str(result["Probability"]) + "%",
                styles["Normal"]
            )
        )

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph(
            "Disclaimer: This report is generated by an educational "
            "machine learning project. It is not a medical diagnosis "
            "and must not replace professional medical advice.",
            styles["Normal"]
        )
    )

    doc.build(elements)

    buffer.seek(0)

    return buffer.getvalue()


# ---------------- SIDEBAR ----------------

st.sidebar.markdown("# 🩺 HealthPredict")
st.sidebar.caption("Healthcare Analytics Platform")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "Dashboard",
        "Prediction Center",
        "Model Comparison",
        "Prediction History",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Project Information")
st.sidebar.caption("Diabetes Prediction using Machine Learning")
st.sidebar.caption("Model: Random Forest Classifier")

st.sidebar.markdown("---")
st.sidebar.caption("Educational project only")


# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.markdown(
        '<p class="main-title">Dashboard Overview</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Your machine learning healthcare analytics workspace.</p>',
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Model Accuracy", "99.40%")

    with col2:
        metric_card("F1 Score", "99.50%")

    with col3:
        metric_card("Recall", "99.67%")

    with col4:
        metric_card("Total Predictions", len(st.session_state.history))

    st.write("")

    st.markdown(
        '<p class="section-title">Project Analytics</p>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1.4, 1])

    with col1:

        st.markdown("#### Model Performance")

        performance = pd.DataFrame({
            "Model": [
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting"
            ],
            "Accuracy": [98.80, 99.40, 99.30],
            "F1 Score": [98.90, 99.50, 99.42],
            "Recall": [98.91, 99.67, 99.50]
        })

        st.bar_chart(
            performance.set_index("Model"),
            y=["Accuracy", "F1 Score", "Recall"]
        )

    with col2:

        st.markdown("#### Dataset Information")

        try:
            data = pd.read_csv("Diabetes_Prediction_Dataset.csv")

            st.metric("Total Records", len(data))
            st.metric("Total Features", len(data.columns) - 1)

            if "Diabetes" in data.columns:

                st.markdown("#### Target Distribution")

                target_count = data["Diabetes"].value_counts()

                st.bar_chart(target_count)

        except FileNotFoundError:

            st.warning(
                "Dataset file not found. Place "
                "Diabetes_Prediction_Dataset.csv "
                "in the project folder to view dataset analytics."
            )

    st.write("")

    st.markdown("#### Welcome to HealthPredict")

    st.markdown("""
    <div class="info-card">
        <h3 style="color:#142B4A;">Intelligent Diabetes Prediction System</h3>
        <p style="color:#718096;">
        This application uses a Random Forest Classification model
        to generate diabetes predictions based on patient health
        and lifestyle information.
        </p>
        <p style="color:#087E8B;">
        Select Prediction Center from the sidebar to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Model metrics are based on the project's test-set evaluation. "
        "They do not guarantee real-world medical performance."
    )


# ---------------- PREDICTION CENTER ----------------

elif page == "Prediction Center":

    st.markdown(
        '<p class="main-title">Patient Assessment</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Enter patient details to generate a model prediction.</p>',
        unsafe_allow_html=True
    )

    with st.form("patient_form"):

        st.markdown("### Personal Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            Age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=30
            )

        with col2:
            Gender = st.selectbox(
                "Gender",
                ["Male", "Female", "Unknown"]
            )

        with col3:
            Family_History = st.selectbox(
                "Family History",
                ["No", "Yes"]
            )

        st.markdown("---")
        st.markdown("### Medical Measurements")

        col1, col2 = st.columns(2)

        with col1:

            BMI = st.number_input(
                "BMI",
                min_value=0.0,
                max_value=100.0,
                value=25.0
            )

            Glucose = st.number_input(
                "Glucose Level",
                min_value=0.0,
                value=100.0
            )

            Blood_Pressure = st.number_input(
                "Blood Pressure",
                min_value=0,
                value=120
            )

        with col2:

            Cholesterol = st.number_input(
                "Cholesterol",
                min_value=0,
                value=180
            )

            Insulin = st.number_input(
                "Insulin",
                min_value=0,
                value=100
            )

            HbA1c = st.number_input(
                "HbA1c",
                min_value=0.0,
                value=5.5
            )

        st.markdown("---")
        st.markdown("### Lifestyle Information")

        col1, col2 = st.columns(2)

        with col1:

            Smoking_Status = st.selectbox(
                "Smoking Status",
                ["Current", "Never", "Former"]
            )

        with col2:

            Physical_Activity = st.selectbox(
                "Physical Activity",
                ["High", "Low", "Moderate"]
            )

        submitted = st.form_submit_button(
            "Generate Prediction",
            use_container_width=True
        )

    if submitted:

        gender_value = {
            "Male": 0,
            "Female": 1,
            "Unknown": 2
        }[Gender]

        smoking_value = {
            "Current": 0,
            "Never": 1,
            "Former": 2
        }[Smoking_Status]

        activity_value = {
            "High": 0,
            "Low": 1,
            "Moderate": 2
        }[Physical_Activity]

        family_value = {
            "No": 0,
            "Yes": 1
        }[Family_History]

        input_data = pd.DataFrame({
            "Age": [Age],
            "Gender": [gender_value],
            "BMI": [BMI],
            "Glucose": [Glucose],
            "Blood_Pressure": [Blood_Pressure],
            "Cholesterol": [Cholesterol],
            "Insulin": [Insulin],
            "HbA1c": [HbA1c],
            "Smoking_Status": [smoking_value],
            "Physical_Activity": [activity_value],
            "Family_History": [family_value]
        })

        prediction = model.predict(input_data)

        probability = None

        if hasattr(model, "predict_proba"):
            probability = round(
                model.predict_proba(input_data)[0][1] * 100,
                2
            )

        if prediction[0] == 1:
            prediction_text = "Diabetes"
        else:
            prediction_text = "No Diabetes"

        result = {
            "Date": datetime.now().strftime("%d-%m-%Y %H:%M"),
            "Age": Age,
            "Gender": Gender,
            "BMI": BMI,
            "Glucose": Glucose,
            "Blood Pressure": Blood_Pressure,
            "Cholesterol": Cholesterol,
            "Insulin": Insulin,
            "HbA1c": HbA1c,
            "Smoking Status": Smoking_Status,
            "Physical Activity": Physical_Activity,
            "Family History": Family_History,
            "Prediction": prediction_text,
            "Probability": probability
        }

        st.session_state.latest_prediction = result
        st.session_state.history.append(result)

    if st.session_state.latest_prediction is not None:

        result = st.session_state.latest_prediction

        st.markdown("---")
        st.markdown("### Prediction Report")

        col1, col2 = st.columns(2)

        with col1:

            if result["Prediction"] == "Diabetes":
                st.error("Model Prediction: Diabetes")
            else:
                st.success("Model Prediction: No Diabetes")

        with col2:

            if result["Probability"] is not None:
                st.metric(
                    "Model-estimated Diabetes Probability",
                    str(result["Probability"]) + "%"
                )

        st.caption(
            "The probability is the model's estimated output, "
            "not a clinically validated risk score."
        )

        st.markdown("#### Patient Details")

        details = pd.DataFrame({
            "Parameter": [
                "Age",
                "Gender",
                "BMI",
                "Glucose",
                "Blood Pressure",
                "Cholesterol",
                "Insulin",
                "HbA1c",
                "Smoking Status",
                "Physical Activity",
                "Family History"
            ],
            "Value": [
                result["Age"],
                result["Gender"],
                result["BMI"],
                result["Glucose"],
                result["Blood Pressure"],
                result["Cholesterol"],
                result["Insulin"],
                result["HbA1c"],
                result["Smoking Status"],
                result["Physical Activity"],
                result["Family History"]
            ]
        })

        st.dataframe(
            details,
            use_container_width=True,
            hide_index=True
        )

        pdf_file = create_pdf(result)

        st.download_button(
            "Download Prediction Report (PDF)",
            data=pdf_file,
            file_name="Diabetes_Prediction_Report.pdf",
            mime="application/pdf"
        )


# ---------------- MODEL COMPARISON ----------------

elif page == "Model Comparison":

    st.markdown(
        '<p class="main-title">Model Performance</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Comparison of the three classification algorithms.</p>',
        unsafe_allow_html=True
    )

    comparison = pd.DataFrame({
        "Model": [
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting"
        ],
        "Accuracy": [98.80, 99.40, 99.30],
        "F1 Score": [98.90, 99.50, 99.42],
        "Recall": [98.91, 99.67, 99.50]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Performance Chart")

    st.bar_chart(
        comparison.set_index("Model"),
        y=["Accuracy", "F1 Score", "Recall"]
    )

    st.success(
        "Random Forest was selected based on the evaluation "
        "results recorded in the project notebook."
    )

    st.info(
        "The displayed scores are from the notebook's test-set "
        "comparison, not a guarantee of future performance."
    )


# ---------------- PREDICTION HISTORY ----------------

elif page == "Prediction History":

    st.markdown(
        '<p class="main-title">Prediction History</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Review predictions generated during this session.</p>',
        unsafe_allow_html=True
    )

    if len(st.session_state.history) > 0:

        history_data = pd.DataFrame(st.session_state.history)

        st.metric(
            "Total Predictions",
            len(history_data)
        )

        st.dataframe(
            history_data,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history_data.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download History (CSV)",
            data=csv_data,
            file_name="Prediction_History.csv",
            mime="text/csv"
        )

        if st.button("Clear Prediction History"):

            st.session_state.history = []
            st.session_state.latest_prediction = None

            st.rerun()

    else:

        st.info(
            "No predictions available yet. Visit Prediction Center "
            "to generate your first prediction."
        )


# ---------------- ABOUT PROJECT ----------------

elif page == "About Project":

    st.markdown(
        '<p class="main-title">About HealthPredict</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Diabetes Prediction Using Machine Learning</p>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
        <h3 style="color:#142B4A;">Project Overview</h3>
        <p style="color:#718096;">
        HealthPredict is an educational machine learning project
        that uses patient health and lifestyle information to
        generate diabetes predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Technologies Used")

    technologies = pd.DataFrame({
        "Technology": [
            "Python",
            "Pandas",
            "NumPy",
            "Scikit-learn",
            "Streamlit",
            "Joblib"
        ],
        "Purpose": [
            "Programming Language",
            "Data Processing",
            "Numerical Operations",
            "Machine Learning",
            "Web Application",
            "Model Saving and Loading"
        ]
    })

    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Selected Model")

    st.success("Random Forest Classifier")

    st.markdown("### Disclaimer")

    st.warning(
        "This application is developed for educational purposes. "
        "It is not a medical diagnosis system. Predictions should "
        "not be used for medical decisions or replace consultation "
        "with a qualified healthcare professional."
    )

