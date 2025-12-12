import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
with open('best_svc_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

st.title("🔬 Breast Cancer Prediction App")
st.write("Enter the values of the 30 features to predict if the cancer is **Benign (non-cancerous)** or **Malignant (cancerous)**.")

# Feature input
features = [
    "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", "Mean Smoothness",
    "Mean Compactness", "Mean Concavity", "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension",
    "Radius SE", "Texture SE", "Perimeter SE", "Area SE", "Smoothness SE",
    "Compactness SE", "Concavity SE", "Concave Points SE", "Symmetry SE", "Fractal Dimension SE",
    "Worst Radius", "Worst Texture", "Worst Perimeter", "Worst Area", "Worst Smoothness",
    "Worst Compactness", "Worst Concavity", "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"
]

user_input = []
cols = st.columns(4)

for i, feature in enumerate(features):
    with cols[i % 4]:
        val = st.number_input(f"{feature}", min_value=0.0, format="%.3f")
        user_input.append(val)

if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)

    if prediction[0] == 0:
        st.error("🔴 Prediction: Malignant (Cancerous)")

        st.markdown("""
        ### ⚠️ Important Guidance for Malignant Prediction:
        - **Consult a Specialist**: Immediately schedule an appointment with an **oncologist** or a **breast cancer specialist**.
        - **Diagnostic Tests**: The doctor may recommend further tests like biopsy, mammogram, or MRI to confirm the diagnosis.
        - **Treatment Options**:
          - Surgery (lumpectomy or mastectomy)
          - Radiation therapy
          - Chemotherapy
          - Hormone therapy (depending on cancer type)
        - **Precautions**:
          - Avoid alcohol and smoking
          - Eat a balanced diet rich in antioxidants (e.g., fruits, vegetables)
          - Manage stress and ensure good sleep
        - **Home Support**:
          - Join a support group
          - Maintain regular communication with your healthcare provider
          - Focus on physical activity (as guided by your doctor)
        """)

    else:
        st.success("🟢 Prediction: Benign (Non-cancerous)")

        st.markdown("""
        ### ✅ Good News: It's Benign!
        - **What it Means**: Benign tumors are **non-cancerous** and usually not life-threatening.
        - **What to Do Next**:
          - Still consult your doctor (preferably a **breast specialist**) for monitoring and follow-up.
          - Some benign conditions may need minor treatments or just observation.
        - **Precautions & Health Tips**:
          - Perform regular breast self-examinations
          - Get routine checkups and mammograms as advised
          - Maintain a healthy lifestyle: exercise, balanced diet, and low stress
        - **Home Remedies & Diet**:
          - Green tea, turmeric, and foods rich in fiber and omega-3
          - Stay hydrated and reduce processed food intake
        """)
