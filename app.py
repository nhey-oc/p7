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

# 2. Class which describes a single flower measurements
class bank_model(BaseModel):
    # https://fastapi.tiangolo.com/python-types/
    test: Optional[str] = None
    test2 : Union[SomeType, None]


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello you.'}



# 3.
@app.post('/predict')

def predict(json_data: bank_model):

    data_dict = dict(json_data)
    data = pd.DataFrame.from_dict(data_dict)

    prediction = LGBM_model.predict([data.iloc[0]])
    # BE CAREFUL, I think it return only 0 or 1. To verify.
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