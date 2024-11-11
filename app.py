import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import gdown
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Google Drive file IDs for each model
CATARACT_MODEL_PATH = 'cataract_detection_model.h5'
CATARACT_GDRIVE_FILE_ID = '1RSueSXHQ3TsyZIf87X7Gs_tuZkyzSGy4'

BRAIN_TUMOR_MODEL_PATH = 'brain_tumor_classification_model.h5'
BRAIN_TUMOR_GDRIVE_FILE_ID = '1BvUvAZcoxXK_PpsiaZuWPwnM5vHe0Xyg'

# Function to download model if not already downloaded
def load_model(model_path, gdrive_file_id=None):
    if gdrive_file_id:
        if not os.path.exists(model_path):
            url = f'https://drive.google.com/uc?id={gdrive_file_id}'
            gdown.download(url, model_path, quiet=False)
        model = tf.keras.models.load_model(model_path)
    return model

# Load existing models
cataract_model = load_model(CATARACT_MODEL_PATH, CATARACT_GDRIVE_FILE_ID)
brain_tumor_model = load_model(BRAIN_TUMOR_MODEL_PATH, BRAIN_TUMOR_GDRIVE_FILE_ID)

# Class labels for brain tumor model
BRAIN_TUMOR_CLASSES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# Load and train the heart disease decision tree model on-the-fly
@st.cache_resource
def train_heart_disease_model():
    heart_data = pd.read_csv('heart.csv')
    X = heart_data.drop(columns=['target'])
    y = heart_data['target']

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Train a decision tree classifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model, scaler

heart_model, heart_scaler = train_heart_disease_model()

# Load and train the diabetes Naive Bayes model on-the-fly
@st.cache_resource
def train_diabetes_model():
    diabetes_data = pd.read_csv('diabetes_dataset.csv')
    X = diabetes_data.drop(columns=['Outcome'])
    y = diabetes_data['Outcome']

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Train a Naive Bayes classifier
    model = GaussianNB()
    model.fit(X, y)
    return model, scaler

diabetes_model, diabetes_scaler = train_diabetes_model()

# Function to preprocess image
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Sidebar for navigation
st.sidebar.title("Medical Imaging Detection")
app_mode = st.sidebar.selectbox("Choose a Detection Mode", ["Cataract Detection", "Brain Tumor Detection", "Heart Disease Prediction", "Diabetes Prediction"])

# Cataract Detection Section
if app_mode == "Cataract Detection":
    st.title("Cataract Detection")
    
    uploaded_file = st.file_uploader("Upload an eye image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Eye Image", use_column_width=True)
        
        image = preprocess_image(image, (150, 150))
        
        if st.button("Predict"):
            prediction = cataract_model.predict(image)[0][0]
            confidence = prediction if prediction > 0.5 else 1 - prediction
            predicted_class = "Cataract-negative" if prediction > 0.5 else "Cataract-positive"
            
            st.write(f"Prediction: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}")

# Brain Tumor Detection Section
elif app_mode == "Brain Tumor Detection":
    st.title("Brain Tumor Detection")
    
    uploaded_file = st.file_uploader("Upload an MRI scan", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
        
        image = preprocess_image(image, (224, 224))
        
        if st.button("Predict"):
            predictions = brain_tumor_model.predict(image)
            predicted_class = BRAIN_TUMOR_CLASSES[np.argmax(predictions)]
            confidence = np.max(predictions)
            
            st.write(f"Prediction: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}")

# Heart Disease Prediction Section
elif app_mode == "Heart Disease Prediction":
    st.title("Heart Disease Prediction")
    
    st.write("Please enter the following information:")
    
    # Collecting user input based on heart dataset
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex (1 = Male, 0 = Female)", options=[1, 0])
    cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", options=[1, 0])
    restecg = st.selectbox("Resting ECG results (0-2)", options=[0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", options=[1, 0])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment (0-2)", options=[0, 1, 2])
    ca = st.selectbox("Number of Major Vessels Colored by Flourosopy (0-3)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (1 = Normal; 2 = Fixed Defect; 3 = Reversible Defect)", options=[1, 2, 3])

    # Prepare input data for prediction
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_data = heart_scaler.transform(input_data)  # Scale the input data

    if st.button("Predict"):
        prediction = heart_model.predict(input_data)[0]
        st.write("Prediction: Heart Disease" if prediction == 1 else "Prediction: No Heart Disease")

# Diabetes Prediction Section
elif app_mode == "Diabetes Prediction":
    st.title("Diabetes Prediction")
    
    st.write("Please enter the following information:")
    
    # Collecting user input based on diabetes dataset
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=85)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

    # Prepare input data for prediction
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    input_data = diabetes_scaler.transform(input_data)  # Scale the input data

    if st.button("Predict"):
        prediction = diabetes_model.predict(input_data)[0]
        st.write("Prediction: Diabetes Positive" if prediction == 1 else "Prediction: No Diabetes")
