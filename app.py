import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('D:/Websites/WineQualityPrediction/trained_model.sav','rb'))

def wine_quality_prediction(input_data):

    # changing the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the data as we are predicting the label for only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==1):
        return 'Good Quality Wine'
    else:
        return 'Bad Quality Wine'
    
def main():
    
    st.title('Wine Quality Prediction Web App')

    #getting input data from the user
    fixed_acidity = st.number_input("Fixed acitity of the wine: ")
    volatile_acidity = st.number_input("Volatile acidity of the wine: ")
    citric_acid = st.number_input("Citric acid present in the wine: ")
    residual_sugar = st.number_input("Residual sugar of the wine: ")
    chlorides = st.number_input("Chlorides amount present in the wine: ")
    free_sulfur_dioxide = st.number_input("Free sulphur dioxide present of the wine: ")
    total_sulphur_dioxde = st.number_input("Total sulphur dioxide present in the wine: ")
    density = st.number_input("Density of the wine: ")
    pH = st.number_input("pH of the wine: ")
    sulphates = st.number_input("sulphates present in the wine: ")
    alcohol = st.number_input("alcohol present in the wine: ")

    output = ''

    #submit button
    if st.button("Predict"):
        output = wine_quality_prediction([fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulphur_dioxde,density,pH,sulphates,alcohol])

    st.success(output)


if __name__ == '__main__':
    main()