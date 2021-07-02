import keras
from pathlib import Path
import streamlit as st
from keras.datasets import mnist 
from PIL import Image
import random
import numpy as np
import pandas as pd

(_, _), (x_test, y_test) = mnist.load_data()

model = keras.models.load_model(Path("mlmodels", "mnist_model1"))

if st.button("Get Next Random Digit"):

    random_idx = random.randint(0, x_test.shape[0]-1)
    digit_img = x_test[random_idx, :,:]
    
    st.image(digit_img)
    
    prediction = model.predict(digit_img.reshape([1, 28,28,1]))[0]
    st.text("Predicted Result : {}".format(np.argmax(prediction)))
    st.text("Truth : {}".format(y_test[random_idx]))