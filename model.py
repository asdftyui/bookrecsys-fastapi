import streamlit as st
import pickle

@st.cache_data
def load_model():
    with open("./model/Cat_model.pkl", 'rb') as file:
        model = pickle.load(file)
    
    return model