# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import streamlit as st
# from PIL import Image
# import hashlib

# # Function to hash passwords
# def hash_password(password):
#     return hashlib.sha256(password.encode()).hexdigest()

# # Initialize user database in session state if not present
# if 'users' not in st.session_state:
#     st.session_state.users = {"admin": hash_password("password123")}

# # Authentication function
# def authenticate(username, password):
#     if username in st.session_state.users and st.session_state.users[username] == hash_password(password):
#         return True
#     return False

# # Streamlit App
# st.title("Heart Disease Prediction App")

# # Authentication System
# if "authenticated" not in st.session_state:
#     st.session_state.authenticated = False

# def login_page():
#     st.subheader("Login")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     if st.button("Login"):
#         if authenticate(username, password):
#             st.session_state.authenticated = True
#             st.session_state.username = username
#             st.rerun()
#         else:
#             st.error("Invalid credentials. Please try again.")

# def register_page():
#     st.subheader("Register")
#     new_username = st.text_input("New Username")
#     new_password = st.text_input("New Password", type="password")
#     if st.button("Register"):
#         if new_username in st.session_state.users:
#             st.warning("Username already exists. Choose a different one.")
#         else:
#             st.session_state.users[new_username] = hash_password(new_password)
#             st.success("Registration successful! You can now log in.")

# def welcome_page():
#     st.subheader("Welcome to the Heart Disease Prediction App")
#     option = st.selectbox("Choose an option", ["Login", "Register"])
#     if option == "Login":
#         login_page()
#     else:
#         register_page()

# if not st.session_state.authenticated:
#     welcome_page()
# else:
#     if st.sidebar.button("Logout"):
#         st.session_state.authenticated = False
#         st.rerun()
    
#     st.subheader(f"Welcome, {st.session_state.username}!")
    
#     # Load the CSV data to a Pandas DataFrame
#     heart_data = pd.read_csv('heart_disease_data.csv')

#     # Prepare the data
#     X = heart_data.drop(columns='target', axis=1)
#     Y = heart_data['target']
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

#     # Train the model
#     model = LogisticRegression()
#     model.fit(X_train, Y_train)

#     # Calculate accuracy
#     training_data_accuracy = accuracy_score(Y_train, model.predict(X_train))
#     test_data_accuracy = accuracy_score(Y_test, model.predict(X_test))

#     # Sidebar for user input
#     st.sidebar.header('User Input Features')
#     def user_input_features():
#         age = st.sidebar.slider('Age', 29, 77, 50)
#         sex = st.sidebar.selectbox('Sex', [0, 1])
#         cp = st.sidebar.selectbox('Chest Pain Type', [0, 1, 2, 3])
#         trestbps = st.sidebar.slider('Resting Blood Pressure', 94, 200, 120)
#         chol = st.sidebar.slider('Cholesterol', 126, 564, 200)
#         fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
#         restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
#         thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 60, 202, 150)
#         exang = st.sidebar.selectbox('Exercise Induced Angina', [0, 1])
#         oldpeak = st.sidebar.slider('Oldpeak', 0.0, 6.2, 1.0)
#         slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
#         ca = st.sidebar.selectbox('Number of Major Vessels (0-3)', [0, 1, 2, 3])
#         thal = st.sidebar.selectbox('Thalassemia', [0, 1, 2, 3])
#         return np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

#     input_data = user_input_features()
#     prediction = model.predict(input_data)

#     # Display prediction
#     st.subheader("Prediction Result")
#     if prediction[0] == 0:
#         st.success("This person does not have heart disease.")
#     else:
#         st.error("This person has heart disease.")

#     # Display model performance
#     st.subheader("Model Performance")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("Training Accuracy", f"{training_data_accuracy:.2%}")
#     with col2:
#         st.metric("Test Accuracy", f"{test_data_accuracy:.2%}")

#     # Optional: Display the dataset
#     if st.checkbox("Show Dataset"):
#         st.subheader("Heart Disease Dataset")
#         st.dataframe(heart_data)

#     # Optional: Display an image
#     try:
#         img = Image.open('heart_img.jpg')
#         st.image(img, caption='Heart Health', use_column_width=True)
#     except FileNotFoundError:
#         st.warning("Image not found. Please ensure 'heart_img.jpg' is in the correct directory.")


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
from PIL import Image
import hashlib

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Initialize user database in session state if not present
if 'users' not in st.session_state:
    st.session_state.users = {"admin": hash_password("password123")}

# Authentication function
def authenticate(username, password):
    if username in st.session_state.users and st.session_state.users[username] == hash_password(password):
        return True
    return False

# Streamlit App
st.title("Heart Disease Prediction App")

# Authentication System
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def login_page():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.page = "Prediction"
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again.")

def register_page():
    st.subheader("Register")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    if st.button("Register"):
        if new_username in st.session_state.users:
            st.warning("Username already exists. Choose a different one.")
        else:
            st.session_state.users[new_username] = hash_password(new_password)
            st.success("Registration successful! You can now log in.")

def welcome_page():
    st.subheader("Welcome to the Heart Disease Prediction App")
    option = st.selectbox("Choose an option", ["Login", "Register"])
    if option == "Login":
        login_page()
    else:
        register_page()

if not st.session_state.authenticated:
    welcome_page()
else:
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction", "Recovery"])
    st.session_state.page = page
    
    st.subheader(f"Welcome, {st.session_state.username}!")
    
    if page == "Prediction":
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

        # Sidebar for user input
        st.sidebar.header('User Input Features')
        def user_input_features():
            age = st.sidebar.slider('Age', 29, 77, 50)
            sex = st.sidebar.selectbox('Sex', [0, 1])
            cp = st.sidebar.selectbox('Chest Pain Type', [0, 1, 2, 3])
            trestbps = st.sidebar.slider('Resting Blood Pressure', 94, 200, 120)
            chol = st.sidebar.slider('Cholesterol', 126, 564, 200)
            fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
            restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
            thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 60, 202, 150)
            exang = st.sidebar.selectbox('Exercise Induced Angina', [0, 1])
            oldpeak = st.sidebar.slider('Oldpeak', 0.0, 6.2, 1.0)
            slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
            ca = st.sidebar.selectbox('Number of Major Vessels (0-3)', [0, 1, 2, 3])
            thal = st.sidebar.selectbox('Thalassemia', [0, 1, 2, 3])
            return np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

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
            st.image(img, caption='Heart Health', use_column_width=True)
        except FileNotFoundError:
            st.warning("Image not found. Please ensure 'heart_img.jpg' is in the correct directory.")
    
    elif page == "Recovery":
        st.subheader("Recovery and Health Tips")
        st.write("Here you can find recovery plans and health tips for heart disease prevention and management.")
        st.markdown("- Maintain a healthy diet rich in vegetables, fruits, and whole grains.")
        st.markdown("- Exercise regularly and stay physically active.")
        st.markdown("- Monitor blood pressure and cholesterol levels.")
        st.markdown("- Avoid smoking and limit alcohol consumption.")
        st.markdown("- Manage stress through meditation and relaxation techniques.")
