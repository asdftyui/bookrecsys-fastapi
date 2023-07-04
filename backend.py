from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from predict import get_prediction
from model import load_model

app = FastAPI()

class UserInfo(BaseModel):
    country: str
    state: str
    city: str
    age: str
    pos_isbn_list: list
    neg_isbn_list: list


@app.post("/predict")
def predict(userinfo: UserInfo):
    model = load_model()
    result = get_prediction(model, userinfo.country, userinfo.state, userinfo.city, 
                             userinfo.age, userinfo.pos_isbn_list+userinfo.neg_isbn_list, len(userinfo.pos_isbn_list))
    result = result[~result['isbn'].isin(userinfo.pos_isbn_list + userinfo.neg_isbn_list)]
    result.sort_values('rating_prediction', ascending=False, inplace=True)
    
    data = {
        'isbn': result.head()['isbn'].values.tolist()
    }
    
    print(data)
    
    return JSONResponse(content=jsonable_encoder(data))

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8001)