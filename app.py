import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import  LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model 

ChestPainType_le = LabelEncoder()
RestingECG_le = LabelEncoder()
ExerciseAngina_le = LabelEncoder()
ST_Slope_le = LabelEncoder()
Sex_le = LabelEncoder()

ChestPainType_le.classes_ = np.load('Encodings/ChestPainType.npy', allow_pickle = True)
RestingECG_le.classes_ = np.load('Encodings/RestingECG.npy', allow_pickle = True)
ExerciseAngina_le.classes_ = np.load('Encodings/ExerciseAngina.npy', allow_pickle = True)
ST_Slope_le.classes_ = np.load('Encodings/ST_Slope.npy', allow_pickle = True)
Sex_le.classes_ = np.load('Encodings/Sex.npy', allow_pickle = True)

model = load_model('Model')

st.header("Heart Failure Prediction")
st.text_input("Enter your Name: ", key="name")


data = pd.read_csv("heart.csv")
data.pop('HeartDisease')
if st.checkbox('Show dataframe'):
    data

st.subheader("Please select relevant medical information:")
#Discrete: Sex, ChestPainType, FastingBS, RestingECG, ExerciseAngina, ST_Slope
#Continuous: Age, RestingBP, Cholesterol, MaxHR, OldPeak, 
left_column, right_column = st.columns(2)
with left_column:
    inp_Sex = st.radio(
        'Sex:',
        np.unique(data['Sex'])
        )
    inp_ChestPainType = st.radio(
        'Chest Pain Type:',
        np.unique(data['ChestPainType'])
    )
    inp_FastingBS = st.radio(
        'Fasting Blood Sugar(1, If greater than 120 mg/dl, 0 otherwise):',
        np.unique(data['FastingBS'])
    )
    inp_RestingECG = st.radio(
        'Resting ElectroCardioGram Results:',
        np.unique(data['RestingECG'])
    )
    inp_ExerciseAngina = st.radio(
        'Exercise Induced Angina:',
        np.unique(data['ExerciseAngina'])
    )
    inp_ST_Slope = st.radio(
        'Slope of peak exercise ST segment:',
        np.unique(data['ST_Slope'])
    )

inp_Age = st.slider('Age(years)', 0, 100)
inp_RestingBP = st.slider('Resting Blood Pressure(mm Hg)', 0, max(data['RestingBP']))
inp_Cholesterol = st.slider('Serum Cholesterol(mm/dl)', 0, 700)
inp_MaxHR = st.slider('Maximum Heart Rate Achieved(/min)', 0, 250)
inp_Oldpeak = st.slider('OldPeak: ST Depression', -3.0, 7.0)

if st.button('Make Prediction'):
    inp_Sex = Sex_le.transform(np.expand_dims(inp_Sex, -1))
    inp_ChestPainType = ChestPainType_le.transform(np.expand_dims(inp_ChestPainType, -1))
    inp_RestingECG = RestingECG_le.transform(np.expand_dims(inp_RestingECG, -1))
    inp_ExerciseAngina = ExerciseAngina_le.transform(np.expand_dims(inp_ExerciseAngina, -1))
    inp_ST_Slope = ST_Slope_le.transform(np.expand_dims(inp_ST_Slope, -1))
    
    inputs = np.expand_dims(
        [inp_Age, int(inp_Sex), int(inp_ChestPainType), inp_RestingBP, inp_Cholesterol, int(inp_FastingBS), 
        int(inp_RestingECG), inp_MaxHR, int(inp_ExerciseAngina), inp_Oldpeak, int(inp_ST_Slope)],
        0
        )
    prediction = model.predict(inputs)
    st.write(f"Heart Failure Risk: ", prediction[0][0] * 100.0, "%")