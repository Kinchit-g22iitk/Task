import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from keras.datasets import mnist

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
session_state = st.session_state

st.markdown("<h1 style='text-align: center;'>Creating your own dataset</h1>", unsafe_allow_html=True)

st.write("Now that you know how computers see and learn, let's talk about something super exciting—creating your own dataset! A dataset is like a big collection of special pictures and information that helps the computer get even smarter. It's a bit like building a treasure chest filled with all the things you want your computer to recognize. Just like how we learn from books, computers learn from these special datasets. The more pictures and details you collect, the better your computer becomes at understanding and recognizing different things. Imagine you're teaching your computer about animals, and your dataset is like a fantastic zoo of pictures! So, creating your own dataset is like being a superhero trainer for your computer, helping it become super smart and learn things in a way that's just right for you.")

# st.title('MNIST')
st.markdown("<h1 style='text-align: center;'>MNIST</h1>", unsafe_allow_html=True)

st.write("Now, let's dive into the world of real machine learning magic! We're going to import something called the MNIST dataset. Imagine MNIST as a giant box of handwritten numbers—like a collection of magic symbols that our computer will learn to recognize. It's like having a box full of numbers written by different people, and our computer will become a number wizard by learning from all of them. This special dataset is a fantastic starting point for our machine learning adventure because it helps our computer understand the shapes and patterns of numbers. So, when we import the MNIST data, it's like opening a treasure chest of handwritten secrets that will turn our computer into a super smart number detective!")
@st.cache_data
def load_data():

    mnist = fetch_openml('mnist_784', as_frame=False, cache=False, parser= 'auto')

    x = mnist.data.astype('float32')
    y = mnist.target.astype('int64')

    x /= 255.0

    return x, y

x, y = load_data()
#For middle part of page

col1, col2= st.columns([40,60])
with col1:
    no_of_samples = st.slider("No. Of Samples",min_value = 0,max_value = 10,value = 9)
    # image = Image.open('C:\Users\HP\OneDrive - IIT Kanpur\3 SEM\project\pages\for_page_2_data.png')
    image = Image.open(r'pages\for_page_2_data.png')

    st.image(image)
indexes = []
for i in range(no_of_samples):
    index = [index for index, value in enumerate(y) if value == i]
    indexes.append(index)
indexes = sum(indexes, [])
x_new = x[indexes]
y_new = y[indexes]

        # Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(x_new)
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_new, cmap='viridis', alpha=0.5)
ax.set_title('Scatter Plot of Modified MNIST Dataset')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
legend = ax.legend(*scatter.legend_elements(), title='Classes')
ax.add_artist(legend)

with col2:
    
    # st.button("VISUALIZE DATA", on_click = display_data(no_of_samples, x, y))
    button_pressed = st.button("VISUALIZE DATA")
    if button_pressed:
        st.pyplot(fig)

st.title('Train-Test Split')
st.write("Now that we have our magical MNIST dataset loaded, it's time to train our computer wizard to recognize those handwritten numbers! But, here's a smart trick: we don't want to test our wizard on the same numbers it learned from; that would be too easy, right? So, we're going to do something called a 'Train-Test Split.' It's like having two sets of magical scrolls—one set to train our computer wizard and another set to test how well it learned. We'll use one set to teach the computer the secrets of handwritten numbers (that's the training part), and then, we'll test its magic on a completely different set of numbers (that's the testing part). This way, we make sure our computer wizard is a true master of recognizing any handwritten number, not just the ones it memorized!")

col21, col22, col23 = st.columns([40,40,20])
with col21:
    data_samples = st.slider('DATA SAMPLES',max_value= 70000,value = 20000,step=1000)

with col22:
    split_ratio = st.slider('SPLIT RATIO',min_value=0.0,max_value=1.0,value=0.7,step = 0.05)
with col23:
    b = st.button("GENERATE")
# if b:

train_data_ = data_samples*split_ratio
train_data_ = int(train_data_)
train_ = f"{int(train_data_)} examples"
test_ = f"{int(data_samples-train_data_)} examples"
data = {
    'Train_data':[train_],
    'Test_data':[test_]
}
# st.beta_set_page_config( layout='wide')
session_state.df = pd.DataFrame(data)
centered_html = f"<div style='display: flex; justify-content: center;'>{session_state.df.to_html(index=False)}</div>"
st.markdown(centered_html, unsafe_allow_html=True)


# st.table(df)
# st.dataframe(session_state.df, hide_index=True)
x = x.reshape(70000,28,28,1)
session_state.train_data_x = x[:train_data_]
session_state.train_data_y = y[:train_data_]
session_state.test_data_x = x[train_data_:(data_samples-train_data_)]
session_state.test_data_y = y[train_data_:(data_samples-train_data_)]


session_state.train_data_y = to_categorical(session_state.train_data_y, num_classes=10)
session_state.test_data_y = to_categorical(session_state.test_data_y, num_classes=10)





