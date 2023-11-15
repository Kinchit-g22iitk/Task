# import page_3
# import page_2
# from pages.3_Model import *
from pages._3_Model import *

from pages._2_Dataset import *
import keras
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, optimizers
from keras.optimizers import Adam
session_state = st.session_state


st.markdown(
    
f"""
        <div style="background-color: #ddd; width: 150px; height: 400px; border-radius: 5px; margin: 5px; overflow: hidden;">
            <div style="background-color: #4CAF50; height: 100%; width: 100%;"></div>
        </div>
    """,
    unsafe_allow_html=True

)
# st.title("Train your model")
st.markdown("<h1 style='text-align: center;'>Train your model</h1>", unsafe_allow_html=True)

st.write("Now that we've designed our magical neural network, it's time to unleash its learning potential through a process known as training. Training is like sending our computer wizard to a school of magic, where it learns to recognize the intricate patterns in our handwritten numbers. We take our enchanted dataset, split into a training set and a testing set, and let our wizard study and understand the nuances. The wizard adjusts its parameters, like a diligent student refining its skills, until it becomes proficient in deciphering the secrets hidden within the numbers. This training process ensures that our model becomes a reliable sorcerer, capable of recognizing not only the patterns it learned but also unseen magical symbols.")

# st.title("Train the network")
st.markdown("<h1 style='text-align: center;'>Train the network</h1>", unsafe_allow_html=True)

st.write("As we embark on the magical journey of training our neural network, envision each iteration as a spellcasting session, refining our wizard's ability to recognize handwritten numbers. The incantations are the mathematical adjustments of our model's parameters, performed during each training epoch. It's like tuning the strings of a magical instrument until the melody is perfect. With each pass through the dataset, our wizard becomes more adept at discerning the unique patterns and features, refining its magical abilities. It's essential to monitor the training process closely, ensuring our wizard doesn't overlearn or underlearn, striking a balance to achieve optimal enchantment. So, let the training commence, and witness as our neural network transforms into a true maestro of recognizing the mystical symbols within our enchanted dataset!")
session_state.epochs_list = []

def train(_model, epochs, lr):
    optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    _model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    accuracy, loss = [], []
    epoch_list = []  

    for epoch in range(epochs):
        st.write(f"Epoch {epoch + 1}")
        progress_bar = st.progress(0.0)

        # Fit the model for one epoch
        history = _model.fit(session_state.train_data_x, session_state.train_data_y, epochs=1, verbose=0)

        # Update the progress bar
        for epoch_step in range(100):
            progress_percentage = (epoch_step + 1) / 100
            progress_bar.progress(progress_percentage)
            time.sleep(0.1)

        # Retrieve and round accuracy and loss values
        epoch_accuracy = round(history.history['accuracy'][0], 2)
        epoch_loss = round(history.history['loss'][0], 2)

        # Store accuracy and loss values
        accuracy.append(epoch_accuracy)
        loss.append(epoch_loss)
        # epoch_list.append(epoch + 1)  

        st.write(f"Training Accuracy: {(100 * epoch_accuracy)}%  Training Loss: {epoch_loss}")
    # session_state.epochs_list = epoch_list
    return accuracy, loss
st.header("TRAIN PARAMETERS")
col31, col32 = st.columns([60,40], gap = 'large')

# st.write(f"{(session_state.model).summary}")
with col31:
    session_state.epochs = st.slider("EPOCHS",min_value=1, value=3,step=1, max_value=10)
    session_state.lr = st.slider("LEARNING RATE" ,min_value=0.0001, value=0.03,max_value=0.3,step=0.0001)
for i in range(session_state.epochs):
    (session_state.epochs_list).append(i+1)
with col32:
    button = st.button("TRAIN MODEL")
session_state.accuracy = []
session_state.loss = []
if button:
    # Generate a timestamp for unique model filename
    session_state.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Append timestamp to the model filename
    model_filename = f"testmodel_{session_state.timestamp}.h5"

    # Train the model
    session_state.accuracy, session_state.loss = train(session_state.model, session_state.epochs, session_state.lr)

    # Save the model with a unique filename
    (session_state.model).save(model_filename)
