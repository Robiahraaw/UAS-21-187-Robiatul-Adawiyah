import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu
import numpy as np

#navigasi sidebar
# horizontal menu
selected2 = option_menu(None, ["Data", "Preprocessing", "Modelling", 'Implementasi'], 
    icons=['house', 'cloud-upload', "list-task", 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal")
selected2

if (selected2 == 'Data') :
    st.title('Deskripsi Data')
    st.write("Data ini Berasal dari Kaggle, berikut saya lampirkan link asal data : https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data")
    st.write("Jumlah fitur pada data ini ada 6 kolom Bertipe Numerik dan 1 Kolom label bertipe data Categorical.")
    st.write("Berikut Deskripsi setiap Fitur :")
    st.write("Age: Usia dalam tahun ketika seorang wanita hamil.")
    st.write("SystolicBP: Nilai atas Tekanan Darah dalam mmHg.")
    st.write("DiastolicBP: Nilai Tekanan Darah yang lebih rendah dalam mmHg.")
    st.write("BS: Kadar glukosa darah dinyatakan dalam konsentrasi molar, mmol/L.")
    st.write("BodyTemp: Suhu Badan dalam Fahrenheit.")
    st.write("HeartRate: Detak jantung istirahat normal dalam denyut per menit.")
    st.write("RiskLevel: Prediksi Tingkat Intensitas Risiko selama kehamilan.")
    st.write("Data ini merupakan data kesehatan ibu hamil, yang di klasifikasi menjadi 3 label yaitu High Risk (Resiko Tinggi), Mid Risk (Resiko Sedang), dan Low Risk (Resiko Rendah).")
    data = pd.read_csv('maternal.csv')
    st.dataframe(data)
if selected2 == 'Preprocessing':
    st.title('Hasil Preprocessing Data')
    st.write("Pada Tahap Preprocessing Data saya menggunakan StandardScaler")
    # Load preprocessed data
    X_td = pd.read_csv('preprocessed_data.csv')
    st.dataframe(X_td)
if (selected2 == 'Modelling') :
    st.title('Hasil Modelling')
    pilih = st.radio('Pilih', ('Naive Bayes', 'Decision Tree', 'KNN', 'ANN'))

    if (pilih == 'Naive Bayes'):
        st.title(' Nilai Akurasi 65%')
    elif (pilih == 'KNN'):
        st.title(' Nilai Akurasi 71,6%')
    elif (pilih == 'Decision Tree'):
        st.title(' Nilai Akurasi 59%')
    elif (pilih == 'ANN'):
        st.title(' Nilai Akurasi 66%')

if (selected2 == 'Implementasi') :
    st.title('Implementasi Maternal Risk Menggunakan Metode KNN')
    # Load the trained model using pickle
    model_filename = 'maternal-knn.pkl'
    with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)

    # Function to predict the RiskLevel from user input
    def predict_risk_level(input_data):
        input_data_scaled = scaler.transform(input_data)  # Perform scaling on input_data
    
        # Reshape input_data_scaled to 2D
        input_data_scaled_2d = input_data_scaled.reshape(1, -1)

        # Perform prediction using the loaded model
        predicted_risk_level = loaded_model.predict(input_data_scaled_2d)

        return predicted_risk_level

    # Input manual from the user
    age = st.number_input('Age')
    systolic_bp = st.number_input('Systolic Blood Pressure')
    diastolic_bp = st.number_input('Diastolic Blood Pressure')
    bs = st.number_input('Blood glucose levels')
    body_temp = st.number_input('Body Temperature')
    heart_rate = st.number_input('Heart Rate')

    # Load the scaler and fit it on a dummy dataset
    scaler = StandardScaler()
    scaler.fit(np.zeros((1, 6)))  # Fit the scaler on a dummy dataset with 1 row and 6 columns

    if st.button('Prediksi'):
        input_user = np.array([age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]).reshape(1, -1)
    
        # Perform scaling on the input using the fitted scaler
        input_user_scaled = scaler.transform(input_user)

        predicted_risk = predict_risk_level(input_user_scaled)
        st.success (f'Predicted Risk Level: {predicted_risk[0]}')