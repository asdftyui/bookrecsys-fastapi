import pandas as pd
import pickle
import streamlit as st


@st.cache_data
def load_books_data():
    path = './data/'
    books = pd.read_csv(path+'books.csv')
    
    return books

@st.cache_data
def load_train_books_data():
    path = './data/'
    books = pd.read_csv(path+'books_catboost.csv')
    
    return books


def book_list():
    return load_books_data()[['isbn', 'book_title']]


def age_map(x) -> int:
    x = int(x)

    if x < 10:
        return 10
    elif x >= 10 and x < 20:
        return 20
    elif x >= 20 and x < 30:
        return 30
    elif x >= 30 and x < 35:
        return 35
    elif x >= 35 and x < 40:
        return 40
    elif x >= 40 and x < 50:
        return 50
    else:
        return 100
    

def users_data_preprocessing(country, state, city):
    path = './data/'
    
    with open(path+'country2idx.pkl', 'rb') as file:
        country2idx = pickle.load(file)
    
    try:
        country_idx = country2idx[country]
    except:
        country_idx = len(country2idx)
    
    with open(path+'state2idx.pkl', 'rb') as file:
        state2idx = pickle.load(file)
    
    try:
        state_idx = state2idx[state]
    except:
        state_idx = len(state2idx)
    
    with open(path+'city2idx.pkl', 'rb') as file:
        city2idx = pickle.load(file)
    
    try:
        city_idx = city2idx[city]
    except:
        city_idx = len(city2idx)
        
        
    return city_idx, state_idx, country_idx
    