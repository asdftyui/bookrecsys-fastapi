from data import users_data_preprocessing, age_map, load_train_books_data
import numpy as np
import pandas as pd

def train_model(model, user_id, location_idx, age_idx, isbn_list, len_pos, books):
   train_input = books[books['isbn'] == isbn_list[0]]
   for i in range(len(isbn_list)-1):
      train_input = pd.concat([train_input, books[books['isbn'] == isbn_list[i+1]]])
   
   train_input['user_id'] = user_id
   train_input['location_city'] = str(location_idx[0])
   train_input['location_state'] = str(location_idx[1])
   train_input['location_country'] = str(location_idx[2])
    
   train_input['years'] = train_input['years'].astype(int)
   train_input['fix_age'] = int(age_idx)
    
   train_input = train_input.dropna()
    
   train_input = train_input[['user_id', 'isbn', 'book_title', 'book_author', 'publisher',
       'language', 'category_high', 'years', 'location_city', 'location_state',
       'location_country', 'fix_age']]
      
   ratings = np.array([10]*len_pos + [1]*(len(isbn_list)-len_pos))
   model.fit(train_input, ratings)
   
   return model


def get_prediction(model, country, state, city, age, isbn_list, len_pos):
    books = load_train_books_data()
   
    user_id = str(float(0))    
    location_idx = users_data_preprocessing(country, state, city)
    age_idx = age_map(age)
    
    model = train_model(model, user_id, location_idx, age_idx, isbn_list, len_pos, books)
    
    test_input = books.copy()
    test_input['user_id'] = user_id
    test_input['location_city'] = str(location_idx[0])
    test_input['location_state'] = str(location_idx[1])
    test_input['location_country'] = str(location_idx[2])
    
    test_input['years'] = test_input['years'].astype(int)
    test_input['fix_age'] = int(age_idx)
    
    test_input = test_input.dropna()
    
    test_input = test_input[['user_id', 'isbn', 'book_title', 'book_author', 'publisher',
       'language', 'category_high', 'years', 'location_city', 'location_state',
       'location_country', 'fix_age']]
    
    prediction = model.predict(test_input)
   
    result = books.dropna().copy()
    result['rating_prediction'] = prediction
    
    return result