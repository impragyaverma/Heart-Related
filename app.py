import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
from PIL import Image

# Load the CSV data to a Pandas DataFrame
heart_data = pd.read_csv('heart_disease_data.csv')

# Prepare the data
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Calculate accuracy
training_data_accuracy = accuracy_score(Y_train, model.predict(X_train))
test_data_accuracy = accuracy_score(Y_test, model.predict(X_test))

# Web app
st.title('Heart Disease Prediction Model')
st.write("This application predicts the presence of heart disease based on user input features.")

# Sidebar for user input
st.sidebar.header('User Input Features')
def user_input_features():
    # Age input
    age = st.sidebar.slider('Age (years)', 29, 77, 50, help="Select your age in years.")
    
    # Sex input - map to numerical values
    sex = st.sidebar.selectbox('Sex', ['Female', 'Male'], help="Select your gender.")
    sex_value = 0 if sex == 'Female' else 1
    
    # Chest Pain Type input - map to numerical values
    cp = st.sidebar.selectbox('Chest Pain Type', 
                             ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'], 
                             help="Select the type of chest pain experienced.")
    cp_value = ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'].index(cp)
    
    # Resting Blood Pressure input
    trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 94, 200, 120, help="Select your resting blood pressure.")
    
    # Cholesterol input
    chol = st.sidebar.slider('Cholesterol (mg/dl)', 126, 564, 200, help="Select your cholesterol level.")
    
    # Fasting Blood Sugar input
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'], help="Is your fasting blood sugar greater than 120 mg/dl?")
    fbs_value = 0 if fbs == 'No' else 1
    
    # Resting Electrocardiographic Results input
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', 
                                  ['Normal', 'Having ST-T wave abnormality', 'Showing probable or definite left ventricular hypertrophy'], 
                                  help="Select the results of your resting electrocardiogram.")
    restecg_value = ['Normal', 'Having ST-T wave abnormality', 'Showing probable or definite left ventricular hypertrophy'].index(restecg)
    
    # Maximum Heart Rate Achieved input
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved (bpm)', 60, 202, 150, help="Select your maximum heart rate achieved.")
    
    # Exercise Induced Angina input
    exang = st.sidebar.selectbox('Exercise Induced Angina', ['No', 'Yes'], help="Do you experience angina induced by exercise?")
    exang_value = 0 if exang == 'No' else 1
    
    # Oldpeak input
    oldpeak = st.sidebar.slider('Oldpeak (depression)', 0.0, 6.2, 1.0, help="Select the oldpeak value.")
    
    # Slope of the Peak Exercise ST Segment input
    slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', 
                                ['Upsloping', 'Flat', 'Downsloping'], 
                                help="Select the slope of the peak exercise ST segment.")
    slope_value = ['Upsloping', 'Flat', 'Downsloping'].index(slope)
    
    # Number of Major Vessels input
    ca = st.sidebar.selectbox('Number of Major Vessels (0-3)', [0, 1, 2, 3], help="Select the number of major vessels colored by fluoroscopy.")
    
    # Thalassemia input
    thal = st.sidebar.selectbox('Thalassemia', 
                               ['Normal', 'Fixed Defect', 'Reversible Defect'], 
                               help="Select the thalassemia status.")
    thal_value = ['Normal', 'Fixed Defect', 'Reversible Defect'].index(thal)
    
    return np.array([[age, sex_value, cp_value, trestbps, chol, fbs_value, restecg_value, 
                     thalach, exang_value, oldpeak, slope_value, ca, thal_value]])

input_data = user_input_features()
prediction = model.predict(input_data)

# Display prediction
st.subheader("Prediction Result")
if prediction[0] == 0:
    st.success("This person does not have heart disease.")
else:
    st.error("This person has heart disease.")

# Display model performance
st.subheader("Model Performance")
col1, col2 = st.columns(2)
with col1:
    st.metric("Training Accuracy", f"{training_data_accuracy:.2%}")
with col2:
    st.metric("Test Accuracy", f"{test_data_accuracy:.2%}")

# Optional: Display the dataset
if st.checkbox("Show Dataset"):
    st.subheader("Heart Disease Dataset")
    st.dataframe(heart_data)

# Optional: Display an image
try:
    img = Image.open('heart_img.jpg')
    st.image(img, caption='Heart Health', use_container_width=True)
except FileNotFoundError:
    st.warning("Image not found. Please ensure 'heart_img.jpg' is in the correct directory.")