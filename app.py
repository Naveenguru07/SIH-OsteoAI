import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(
    page_title="OsteoAI",
    page_icon="🦵",
    layout="wide"
)

# Load model
@st.cache_resource
def load_ai_model():
    return load_model("best_oa_model_v2.keras")

model = load_ai_model()

# Grade information
GRADE_INFO = {
    0: {
        "severity": "Normal",
        "description": "No clear radiographic evidence of osteoarthritis."
    },
    1: {
        "severity": "Doubtful",
        "description": "Possible early signs of osteoarthritis."
    },
    2: {
        "severity": "Mild",
        "description": "Mild osteoarthritic changes detected."
    },
    3: {
        "severity": "Moderate",
        "description": "Moderate osteoarthritic changes detected."
    },
    4: {
        "severity": "Severe",
        "description": "Severe osteoarthritic changes detected."
    }
}

# Header
st.title("🦵 OsteoAI")
st.subheader("AI-Assisted Knee Osteoarthritis Assessment")

st.info(
    "This system provides preliminary AI-based screening from knee X-ray images. "
    "It is not a medical diagnosis."
)

# Patient information
st.header("👤 Patient Information")

col1, col2 = st.columns(2)

with col1:
    patient_name = st.text_input("Patient Name")

with col2:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

# Upload
st.header("📤 Upload Knee X-Ray")

uploaded_file = st.file_uploader(
    "Choose a knee X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded Knee X-Ray",
            use_container_width=True
        )

    with col2:

        st.write("### Image Information")
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Patient:** {patient_name if patient_name else 'Not provided'}")
        st.write(f"**Age:** {age}")

        analyze = st.button(
            "🔍 Analyze X-Ray",
            type="primary",
            use_container_width=True
        )

        if analyze:

            with st.spinner("AI is analyzing the X-ray..."):

                # Preprocessing
                img = image.resize((224, 224))
                img_array = np.array(img, dtype=np.float32)

                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                # Prediction
                predictions = model.predict(
                    img_array,
                    verbose=0
                )

                predicted_class = int(
                    np.argmax(predictions[0])
                )

                confidence = float(
                    np.max(predictions[0]) * 100
                )

                info = GRADE_INFO[predicted_class]

            st.success("Analysis completed successfully!")

            st.header("📊 AI Assessment Result")

            r1, r2, r3 = st.columns(3)

            with r1:
                st.metric(
                    "KL Grade",
                    predicted_class
                )

            with r2:
                st.metric(
                    "Severity",
                    info["severity"]
                )

            with r3:
                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            st.subheader("Assessment")

            st.write(info["description"])

            # Probability details
            st.subheader("📈 Grade Probabilities")

            for grade, probability in enumerate(predictions[0]):
                st.write(
                    f"KL Grade {grade}: "
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    float(probability)
                )

# Disclaimer
st.divider()

st.warning(
    "⚠️ Medical Disclaimer: This application is intended for "
    "educational, research, and preliminary screening purposes only. "
    "It does not provide a medical diagnosis. Please consult a "
    "qualified healthcare professional for clinical evaluation."
)