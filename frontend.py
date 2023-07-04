import streamlit as st
import requests
import pandas as pd
from data import load_books_data, book_list#, users_data_preprocessing


def main(): 
    st.title("Book Rating Prediction Model")
    
    book = book_list()
    
    with st.form(key="user 정보 입력 form"):
        st.write("user 정보를 입력해주세요")
        country = st.text_input("country")
        state = st.text_input("state")
        city = st.text_input("city")
        age = st.text_input("age")
        
        st.write("좋아하는 책을 선택해주세요")
        pos_titles = st.multiselect('favorite books', book['book_title'])
        
        st.write("싫어하는 책을 선택해주세요")
        neg_titles = st.multiselect('disliked books', book['book_title'])
        
        st.form_submit_button("submit")
    
    if not country:
        st.warning("country 정보를 입력해주세요.")
    elif not state:
        st.warning("state 정보를 입력해주세요.")
    elif not city:
        st.warning("city 정보를 입력해주세요.")
    elif not age:
        st.warning("age 정보를 입력해주세요.")
    elif not pos_titles:
        st.warning("좋아하는 책을 선택해주세요.")
    elif not neg_titles:
        st.warning("싫어하는 책을 선택해주세요.")
    else:
        books = load_books_data()
        
        # user가 선호하는 title을 isbn으로 반환
        pos_isbn_list = []
        for title in pos_titles:
            pos_isbn_list.append(book.loc[book['book_title'] == title, 'isbn'].reset_index(drop=True)[0])
        
        # user가 선호하지 않는 title을 isbn으로 반환
        neg_isbn_list = []
        for title in neg_titles:
            neg_isbn_list.append(book.loc[book['book_title'] == title, 'isbn'].reset_index(drop=True)[0])
        
        
        data = {
            'country': country,
            'state': state,
            'city': city,
            'age': age,
            'pos_isbn_list': pos_isbn_list,
            'neg_isbn_list': neg_isbn_list
        }
        
        response = requests.post("http://localhost:8001/predict", json = data)
        isbn = pd.DataFrame(response.json()['isbn'], columns=['isbn'])
        st.write(books.merge(isbn, on='isbn'))
        
        # 입력받은 user 정보로 book rating을 에측해 top-5개의 책 추천
        # result = get_prediction(model, country, state, city, age, pos_isbn_list + neg_isbn_list, len(pos_isbn_list))
        # result = result[~result['isbn'].isin(pos_isbn_list + neg_isbn_list)]
        # result.sort_values('rating_prediction', ascending=False, inplace=True)
        # st.write(books.merge(result.head()['isbn'], on='isbn'))
    
main()