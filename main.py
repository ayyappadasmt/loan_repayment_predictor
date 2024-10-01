import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Streamlit app title
st.title('Loan Repayment Predictor')
st.subheader("created by AYYAPPADAS M.T.")

# Load and prepare the dataset
# Assuming you have a dataset saved as 'loan_data.csv'
df = pd.read_csv('E:\loan repayment predictor\Decision_Tree_ Dataset.csv')  # Replace with your actual dataset path
X = df.values[:, 0:4]
Y = df.values[:, 4]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# Train the Decision Tree Classifier
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

# Get user inputs
st.subheader('Enter Loan Information')
initial_payment = st.number_input('Initial Payment', min_value=0, value=1000)
last_payment = st.number_input('Last Payment', min_value=0, value=1000)
credit_score = st.number_input('Credit Score', min_value=0, value=600)
house_number = st.number_input('House Number', min_value=0, value=1)

# Collect the user inputs in the form of a NumPy array (reshape to match the model's expected input)
user_input = np.array([[initial_payment, last_payment, credit_score, house_number]])

# Make prediction based on user input
if st.button('Predict Loan Repayment'):
    prediction = clf_entropy.predict(user_input)
    
    # Display the prediction
    if prediction[0] == "yes":
        st.success("The loan will be repaid.")
    else:
        st.error("The loan will not be repaid.")

# Optionally: Display the accuracy of the model on the test set
y_pred = clf_entropy.predict(X_test)
accuracy = accuracy_score(y_pred, y_test) * 100
st.write(f"Model accuracy on test data: {accuracy:.2f}%")
