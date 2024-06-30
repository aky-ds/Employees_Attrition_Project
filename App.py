import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
def preprocess_data(df, new_data):
  """
  Preprocesses data by encoding categorical features and scaling numerical features.
  """
  le = LabelEncoder()
  scaler = MinMaxScaler()

  # Encode categorical features
  categorical_features = ['Job Role', 'Work-Life Balance', 'Job Satisfaction', 
                          'Performance Rating', 'Overtime', 'Education Level', 
                          'Marital Status', 'Job Level', 'Remote Work',
                          'Leadership Opportunities', 'Innovation Opportunities',
                          'Company Reputation', 'Employee Recognition']
  for col in categorical_features:
    df[col] = le.fit_transform(df[col])
    new_data[col] = le.transform(np.array([new_data[col]]))

  # Scale numerical features
  numerical_features = ['Monthly_income', 'Company_size', 'Company_tenure']
  df[numerical_features] = scaler.fit_transform(df[numerical_features])
  new_data[numerical_features] = scaler.transform(np.array([new_data[numerical_features]]).reshape(1, -1))

  return df, new_data

# Load model
file1 = open('model.pkl', 'rb')
model = pickle.load(file1)
file1.close()

# Read data
df=pd.read_csv("cleaned_employ_Data.csv")
df.drop("Employee ID",axis=1,inplace=True)
df.drop("Unnamed: 0",axis=1,inplace=True)

# Title and User Input
st.title("Employee Attrition Application Model")

new_data = {}
new_data['Age'] = st.selectbox('Age', df['Age'].unique())
new_data['Gender'] = st.selectbox('Gender', df['Gender'].unique())
new_data['Years_at_com'] = st.selectbox('Years at company', df['Years at Company'].unique())
new_data['Job_Role'] = st.selectbox('Job Role', df['Job Role'].unique())

# Handle monthly income with error handling
try:
  new_data['Monthly_income'] = int(st.text_input("Enter the monthly income"))
except ValueError:
  st.error("Please enter a valid number for monthly income.")
  new_data['Monthly_income'] = np.nan  # Set to NaN for missing value

# ... (similarly collect data for all features with error handling for numerical values)

if st.button('Predict'):
  # Preprocess data
  df_processed, new_data_processed = preprocess_data(df.copy(), new_data.copy())

  # Make prediction
  prediction = model.predict(new_data_processed)[0]
  proba = model.predict_proba(new_data_processed)[0].max()

  # Display prediction result
  if prediction == 1:
    st.success("Employee is likely to leave the company.")
  else:
    st.success("Employee is likely to stay with the company.")
  st.write(f"Prediction Probability: {proba:.2f}")
