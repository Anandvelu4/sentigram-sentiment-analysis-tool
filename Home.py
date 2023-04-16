import streamlit as st
import base64
from PIL import Image
import requests


st.set_page_config(
    page_title=("Sentigram Home Page "),
    page_icon=("https://example.com/icon.jpg")
)
st.title("Sentigram: A Sentiment Analysis Tool")
st.sidebar.success("Select a feature from above")
# Add "About Us" section
st.header("About Us")
st.write("I, Anand and Vibhave  a team of bca students who are passionate about natural language processing and machine learning. Our goal is to build tools that make it easy for anyone to analyze and understand text data.")

# Add image

image = Image.open("C:\\Users\\Anand\\PycharmProjects\\SentigramAVMajor\\static\\project.jpg")

st.image(image, caption=f"<b>Our Team <b>")



