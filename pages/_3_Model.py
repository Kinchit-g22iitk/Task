from pages._2_Dataset import *
import keras
import tensorflow as tf
import streamlit as st
from PIL import Image
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, optimizers
import visualkeras
session_state = st.session_state



# st.table(session_state.df)

# st.title('Building your own model')
st.markdown("<h1 style='text-align: center;'>Building your own model</h1>", unsafe_allow_html=True)

st.write("Now comes the most thrilling part of our journey into the realm of machine learning—building our very own model! Think of a model as a special recipe our computer wizard follows to recognize patterns and make predictions. It's like creating a unique spell that allows our computer to understand the magic in the handwritten numbers of the MNIST dataset. We'll choose the ingredients of our spell, known as parameters, and train our model using the training set we prepared earlier. Just like a master chef perfecting a dish, we'll tweak and adjust our model until it becomes a true expert at recognizing handwritten numbers. Building a model is where the magic happens, and soon, our computer will be ready to show off its newfound skills!")

# st.title('Design a neural network')
st.markdown("<h1 style='text-align: center;'>Design a neural network</h1>", unsafe_allow_html=True)

st.write("Now, let's get creative and design our very own magical neural network! Think of a neural network as a powerful spellbook that helps our computer wizard understand the intricate details of handwritten numbers. We have some exciting choices to make. First, we can choose a Convolutional Neural Network (CNN) model, which is like giving our wizard special lenses to focus on specific features in the numbers. This is perfect for capturing the unique patterns in handwritten digits. Next, we get to decide on activation functions, the secret ingredients that add a touch of magic to each layer of our spellbook. We might choose ReLU for its energetic bursts of magic or Sigmoid for a smoother touch. These activation functions determine how our wizard reacts to the patterns it discovers. So, get ready to weave the threads of your own magical neural network and watch as it transforms our computer into a true sorcerer of handwritten number recognition!")
# st.title("Model Design")
st.markdown("<h1 style='text-align: center;'>Model Design</h1>", unsafe_allow_html=True)

st.write("""Embarking on the journey of designing your very own neural network is both exciting and empowering! Just like crafting a magical spellbook for our computer wizard, the choices we make in model design shape its ability to recognize patterns. Whether you opt for the predefined enchantment of a pre-trained model or embark on the customization adventure, the power is in your hands.

For those choosing the mystical path of customization, consider starting with a Conv2D layer – think of it as granting your wizard special lenses to focus on intricate details. These layers excel at capturing the essence of features in your data, especially beneficial when dealing with images, like the mystical handwritten numbers in the MNIST dataset. Once you've set the foundation with Conv2D, it's time for a MaxPooling layer – a spell that helps your wizard consolidate and magnify the most important information it gathered. It's like giving your wizard a moment to reflect and summarize its newfound insights. Remember, after each Conv2D layer, consider adding a MaxPooling layer to enhance the magical learning process. And for the grand finale, the last layer should be a Dense layer with 10 neurons, ensuring your wizard becomes a true master in deciphering the mystical patterns hidden within your data!

Whether you opt for the tried-and-true pre-trained models or dive into the creation of your own, the magic of neural networks awaits your command!""")
def display_image(model):
    # Generate a layered view of the model with adjusted parameters
    image = visualkeras.layered_view(model)
    st.image(image, caption='Layered View of the Model', use_column_width=True)
def model_creation(selected_layer_type, hidden_units):
    model = Sequential()
    for i in range(len(selected_layer_type)):
        if selected_layer_type[i] == "Conv2D" and i == 0:
            model.add(Conv2D(filters=hidden_units[i][0],kernel_size=(hidden_units[i][1],hidden_units[i][2]), activation='relu',input_shape=(28,28,1),strides = 1, name=f'denseinput_{i}'))
        elif selected_layer_type[i] == "Conv2D":
            model.add(Conv2D(filters=hidden_units[i][0],kernel_size=(hidden_units[i][1],hidden_units[i][2]), activation='relu',strides = 1, name=f'conv2d_{i}'))
        elif selected_layer_type[i] == "Dropout":
            model.add(Dropout(hidden_units[i], name=f'dropout_{i}'))
        elif selected_layer_type[i] == "BatchNormalization":
            model.add(BatchNormalization(name=f'batchnormalization_{i}'))
        elif selected_layer_type[i] == "Flatten":
            model.add(Flatten(name=f'flatten_{i}'))

        elif selected_layer_type[i] == "MaxPooling2D":
            model.add(MaxPooling2D(pool_size=(hidden_units[i][0],hidden_units[i][1]), strides = hidden_units[i][2], name=f'maxpooling2d_{i}'))
        elif selected_layer_type[i] == "Dense" and i == len(selected_layer_type) - 1:
            model.add(Dense(10,activation = 'softmax',name=f'dense_output'))
        elif selected_layer_type[i] == "Dense":
            model.add(Dense(hidden_units[i],activation='relu', name=f'dense_{i}'))

    return model
def build_custom(number):
    
    selected_layer_type = ["Conv2D","BatchNormalization","Conv2D","BatchNormalization","MaxPooling2D","Dropout","Conv2D","BatchNormalization","Conv2D","BatchNormalization","MaxPooling2D","Dropout","Flatten","Dense","BatchNormalization","Dropout","Dense","BatchNormalization","Dropout","Dense"]
    hidden_units = [[32,3,3],'',[32,3,3],'',[2,2,1],0.25,[64,3,3],'',[64,3,3],'', [2,2,1],0.25,'',512,'',0.25,1024,'',0.5,10]
    model = model_creation(selected_layer_type, hidden_units)
    st.write("Model Generated!")
    
    display_image(model)
    return model
def get_layer_input(layer_number):
        layer_type = st.selectbox(f"Select Layer Type {layer_number}", ["Conv2D", "Dense","BatchNormalization","MaxPooling2D","Dropout","Flatten"])
        if layer_type == "Dense":
            hidden_units = st.number_input(f"Enter the number of neurons for {layer_number} dense layer", min_value=1,step=1)
        elif layer_type == "MaxPooling2D":
            poolsize = st.selectbox(f"Select the type of pool_size {layer_number} layer", [[2,2],[3,3]])
            strides = st.selectbox(f"Choose the number of strides for {layer_number} layer",[1,2])
            hidden_units = [poolsize[0],poolsize[1],strides]
        elif layer_type == "BatchNormalization" or layer_type == "Flatten":
            hidden_units = ''
        elif layer_type == "Conv2D":
            filters =  st.slider(f"Enter the number of filters for {layer_number} dense layer", min_value=1,max_value=128)
            kernel_size = st.selectbox(f"Select the type of kernel for {layer_number} layer", [[2,2],[3,3],[4,4]])
            hidden_units = [filters, kernel_size[0], kernel_size[1]]
        elif layer_type == "Dropout":
            hidden_units = st.slider(f"Enter the dropout for {layer_number} layer",min_value = 0.1, step = 0.05, max_value = 1.0)
        return layer_type, hidden_units
def custom():

    # Streamlit app
    st.header("Neural Network Layer Configuration")

    # Initial selection of layer type
    layer_number = 1
    st.header("Select Initial Layer Type:")
    selected_layer, hidden_u = get_layer_input(layer_number)
    selected_layer_type = []
    hidden_units = []
    selected_layer_type.append(selected_layer)
    hidden_units.append(hidden_u)
    # Display selected layer type
    st.write(f"Selected Layer Type: {selected_layer}")
    
    add_more_layers = st.checkbox("Add More Layers")

    # If the user chooses to add more layers
    while add_more_layers:
        layer_number += 1
        st.header(f"Select Additional Layer Type {layer_number}")
        layer_type, hidden_u = get_layer_input(layer_number)
        selected_layer_type.append(layer_type)
        hidden_units.append(hidden_u)
        
        add_more_layers = st.checkbox(f"Add More Layer {layer_number}")
    model = model_creation(selected_layer_type, hidden_units)
    if st.button("Generate Model"):
        
        st.write("Model Generated!")
        
        display_image(model)
    
    return model

session_state.type = st.selectbox(f"Select the type of model which you want for you model",["Pre Trained","Custom"], index=None)
session_state.model = keras.Sequential()
if session_state.type == "Custom":
    model = custom()
    session_state.model = model
    
elif session_state.type == "Pre Trained":
    model = build_custom(1)
    session_state.model = model
    st.write("Model Generated!")

