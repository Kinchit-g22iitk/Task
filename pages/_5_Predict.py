import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os
import numpy as np
import keras 
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
import cv2
import time
from pages._3_Model import *
from pages._2_Dataset import *
from pages._4_Train import *
from datetime import datetime
session_state = st.session_state


# st.title("Make Predictions on your Data")
st.markdown("<h1 style='text-align: center;'>Make Predictions on your Data</h1>", unsafe_allow_html=True)


st.write("Now that our magical neural network has completed its training and is brimming with newfound knowledge, it's time to witness its enchanting abilities in action by making predictions on your very own data. Just like a wizard putting their magical skills to the test, you can feed your network with new, unseen examples and watch as it unveils its proficiency in recognizing patterns. Simply present it with the mystical symbols you want deciphered, and let the magic unfold. The network will provide predictions, offering insights into the patterns it has learned during its training, transforming your computer into a mystical oracle capable of unraveling the secrets hidden within your unique dataset. It's the culmination of your magical journey into machine learning, where your very own neural network becomes a reliable guide in the realm of predictions.")

# st.header("Test Data")
st.markdown("<h1 style='text-align: center;'>Test Data</h1>", unsafe_allow_html=True)

st.write("Welcome to the realm of interactive enchantment under the 'Test Data' banner! Here, you hold the key to unlocking the magic within your own handwritten symbols or even drawing a digit. By offering the ability to upload images for testing, you become the conjurer of queries, seeking insights from our trained neural network. Picture this as your personal crystal ball, where you can present any mystical symbol you desire, and our sorcerer of a neural network will reveal its interpretation. Whether it's a digit you've penned yourself or a symbol from the magical tapestry of your imagination, this feature empowers you to witness firsthand the mystical predictions and interpretations our trained model can unveil. Upload your own images and let the magic unfold as your computer wizard showcases its ability to decipher the secrets within the visual enchantment you provide.")
drawn_images_folder = "drawn_images"
# @st.
@st.cache_data
def pre_trained_model_generation():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    inpx = (28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    inpx = Input(shape=inpx)
    layer1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inpx)
    layer2 = Conv2D(64, (3, 3), activation='relu')(layer1)
    layer3 = MaxPooling2D(pool_size=(3, 3))(layer2)
    layer4 = Dropout(0.5)(layer3)
    layer5 = Flatten()(layer4)
    layer6 = Dense(250, activation='sigmoid')(layer5)
    layer7 = Dense(10, activation='softmax')(layer6)
    model = Model([inpx], layer7)
    model.compile(optimizer=keras.optimizers.Adadelta(),
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
 
    model.fit(x_train, y_train, epochs=8, batch_size=500)
    model_filename = f"default_model.h5"
    model.save(model_filename)
    return 
os.makedirs(drawn_images_folder, exist_ok=True)
pre_trained_model_generation()
def is_digit_drawn(image_data):
        # Convert to grayscale
        gray = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)

        # Apply thresholding
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Count non-zero pixels (if any digit is drawn)
        return cv2.countNonZero(thresh) > 0
col1, col2 = st.columns([50,50], gap="large")
with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file to the drawn_images folder
        drawn_images_folder = "drawn_images"
        os.makedirs(drawn_images_folder, exist_ok=True)
        uploaded_file_path = os.path.join(drawn_images_folder, "uploaded_image.png")
        
        # Use BytesIO to read the file and save it
        uploaded_file.seek(0)
        with open(uploaded_file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    st.markdown("<p style='text-align: center;'>OR</p>", unsafe_allow_html=True)

    canvas_result = st_canvas(
        stroke_width=25,
        stroke_color="#fff",
        background_color="#000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None and is_digit_drawn(canvas_result.image_data):
        # Save the drawn image to the folder
        drawn_image = Image.fromarray(canvas_result.image_data.astype("uint8"))
        drawn_image_path = os.path.join(drawn_images_folder, "drawn_image.png")
        drawn_image.save(drawn_image_path)

        st.success("Digit is drawn and image saved successfully!")


def get_prediction(img_path):
    # model_filename = f"testmodel_{session_state.timestamp}.h5"
    model_filename = f"testmodel_{session_state.timestamp}.h5" if hasattr(session_state, 'timestamp') and session_state.timestamp else "default_model.h5"

    
# img_28x28 = np.array(image.resize((28, 28), Image.ANTIALIAS))
    model = tf.keras.models.load_model(model_filename)

    img = Image.open(img_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # img_array = np.reshape(1,28,28)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_digit = np.argmax(predictions)
    return predicted_digit

def get_image_path():

    # Check if the drawn image exists
    drawn_image_path = os.path.join(drawn_images_folder, "drawn_image.png")
    uploaded_image_path = os.path.join(drawn_images_folder, "uploaded_image.png")
    if os.path.exists(drawn_image_path):
        return drawn_image_path

    # Check if the uploaded image exists
    elif os.path.exists(uploaded_image_path):
        return uploaded_image_path

    # Return a default path or handle the case where no image is available
    return None # Update with your default image path

# Example usage
img_path = get_image_path()
with col2:
    bu = st.button("PREDICT")
    if bu:
        with st.spinner(''):
            time.sleep(5)
            session_state.prediction = get_prediction(img_path)
        st.text_area(label="xyz", value=f"The predicted digit is {session_state.prediction}",label_visibility="hidden",height=1)
            

        # prediction = get_prediction(img_path)
        # st.write(f"The predicted digit is {prediction}")



col51, col52 = st.columns([1,1],gap = "large")


with col51:
    st.write("Is the model Prediction Correct?")

with col52:
    yes = st.button("Yes")
    no = st.button("No")
    if yes:
        st.write("Cool")
    elif no:
        predict = st.button("PREDICT AGAIN")
        if predict:
            with st.spinner(''):
                time.sleep(5)
                session_state.prediction = get_prediction(img_path)
        # st.text_area(label="xyz", value=f"The predicted digit is {session_state.prediction}",label_visibility="hidden",height=1)
