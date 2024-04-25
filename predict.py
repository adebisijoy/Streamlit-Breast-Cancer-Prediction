import numpy as np
import pickle as pk
import streamlit as st 

# loading the saved model
loaded_model = pk.load(open("predict_data.pkl", 'rb'))

# Creating a function for Prediction


def cancer_prediction(input_data):
    # Convert input data to float and handle any errors
    try:
        input_data_numeric = [float(val) for val in input_data]
    except ValueError:
        return "Please enter valid numeric values for all input fields."

    # Reshape the numpy array as we predict for one datapoint
    input_data_reshaped = np.array(input_data_numeric).reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return "The breast is benign"
    else:
        return "The breast is malignant"


def main():

    # giving title
    st.title("Breast Cancer Prediction Web App")

    # getting the input data from the user

    radius_m = st.text_input("Radius_mean value")
    texture = st.text_input("Texture_mean value")
    perimeter = st.text_input("Perimeter_mean value")
    area = st.text_input("Area_mean value")
    smooth = st.text_input("Smoothness_mean value")
    compact = st.text_input("Compactness_mean value")
    concave = st.text_input("Concavity_mean value")
    points = st.text_input("Concave points_mean value")
    symmetry = st.text_input("Symmetry_mean value")
    radius_se = st.text_input("Radius_se value")
    perimeter_se = st.text_input("Perimeter_se value")
    area_se = st.text_input("Area_se value")
    compact_se = st.text_input("Compactness_se value")
    concave_se = st.text_input("Concavity_se value")
    point_se = st.text_input("Concave points_se value")
    radius_worst = st.text_input("Radius_worst value")
    texture = st.text_input("Texture_worst value")
    perimeter_worse = st.text_input("Perimeter_worst")
    area_worst = st.text_input("Area_worst value")
    smooth_worse = st.text_input("Smoothness_worst value")
    compact_worse = st.text_input("Compactness_worst value")
    concave_worse = st.text_input("Concavity_worst value")
    points_worse = st.text_input("Concave points_worst value")
    symmetry_worst = st.text_input("Symmetry_worst value")
    fractal = st.text_input("Fractal_dimension_worst value")

    # code for prediction
    diagnosis = ''

    # creating a button for prediction
    if st.button("Breast Cancer Test Result"):
        diagnosis = cancer_prediction([radius_m, texture, perimeter, area, smooth, compact, concave,
                                       points, symmetry, radius_se, perimeter_se, area_se, compact_se,
                                       concave_se, point_se, radius_worst, texture, perimeter_worse, area_worst, smooth_worse, compact_worse,
                                       concave_worse, points_worse, symmetry_worst, fractal])

    st.success(diagnosis)


if __name__ == '__main__':
    main()
