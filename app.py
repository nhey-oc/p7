# Inspired by https://raw.githubusercontent.com/krishnaik06/FastAPI/main/app.py

# 1. Library imports
import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd

from fastapi.encoders import jsonable_encoder

# 2. Create the app object
app = FastAPI()
with open('models/LGBM_model.pkl', 'rb') as f:
    LGBM_model = pickle.load(f)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello you.'}


# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')

def predict(json_data):
    print('test')
    prediction = LGBM_model.predict([json_data])
    if(prediction[0]>0.5):
        prediction=1
    else:
        prediction=0
    return {
        'prediction': prediction
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn app:app --reload