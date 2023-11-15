import streamlit as st
from PIL import Image
st.markdown("<h1 style='text-align: center;'>Introduction to Image Classification</h1>", unsafe_allow_html=True)



st.write('Image Classification (often referred to as Image Recognition) is the task of associating one (single-label classification) or more (multi-label classification) labels to a given image.In other words, given an image, the goal is to categorize it into one of several classes. So in this interactive GUI,how modern image classification works with a simple example of applying a CNN model for predicting handwritten digit.')
st.title('How Computers See')
# st.markdown("<h1 style='text-align: center;'>Introduction to Image Classification</h1>", unsafe_allow_html=True)
st.write("Imagine your computer having its very own eyes! These electronic eyes are like super-smart cameras that can take pictures and videos of the world. But here's the cool part: they don't just see pictures like we do. They see everything as tiny dots, like the dots in your favorite coloring book. These dots have colors and brightness, and the computer puts them together like a magical puzzle. This helps the computer understand what's in the pictures, just like you recognize your friend's face. This special trick, called 'computer vision,' is like giving your computer the power to understand and learn from the pictures it sees. So, when you hear about machines learning, it's a bit like your computer going to school for its very own superhero eyes!")

image = Image.open(r'pages/for_page_1_data.png')
st.image(image)



