import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Load model
model = pickle.load(open('iris.pkl', 'rb'))
# Set title
st.title("🌸 Iris Flower Prediction App")

st.subheader("Enter the flower measurements:")

sepal_length = st.number_input("Sepal Length", min_value=0.0, format="%.2f")
sepal_width = st.number_input("Sepal Width", min_value=0.0, format="%.2f")
petal_length = st.number_input("Petal Length", min_value=0.0, format="%.2f")
petal_width = st.number_input("Petal Width", min_value=0.0, format="%.2f")

image0 = Image.open("static/setosa.jpg")
image1 = Image.open("static/verci.jpg")
image2 = Image.open("static/flower1.jpg")

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    predicted_class = prediction[0]

    if predicted_class == 0:
        st.image(image0, caption='Setosa 🌸')
        st.success("Predicted: Setosa")
    elif predicted_class == 1:
        st.image(image1, caption='Versicolor 🌼')
        st.success("Predicted: Versicolor")
    elif predicted_class == 2:
        st.image(image2, caption='Virginica 🌺')
        st.success("Predicted: Virginica")
