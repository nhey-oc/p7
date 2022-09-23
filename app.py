# Inspired by https://raw.githubusercontent.com/krishnaik06/FastAPI/main/app.py

# 1. Library imports
import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd

from pydantic import BaseModel
from typing import Union


from fastapi.encoders import jsonable_encoder

# 2. Create the app object
app = FastAPI()
with open('models/LGBM_model.pkl', 'rb') as f:
    LGBM_model = pickle.load(f)

# 2. Class which describes a single flower measurements
class bank_model(BaseModel):
    # https://fastapi.tiangolo.com/python-types/
    #test: Optional[str] = None
    #test2 : Union[SomeType, None]

    index: float
    SK_ID_CURR: float
    CODE_GENDER: float
    FLAG_OWN_CAR: float
    FLAG_OWN_REALTY: float
    CNT_CHILDREN: float
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    REGION_POPULATION_RELATIVE: float
    DAYS_BIRTH: float
    DAYS_REGISTRATION: float
    DAYS_ID_PUBLISH: float
    FLAG_MOBIL: float
    FLAG_EMP_PHONE: float
    FLAG_WORK_PHONE: float
    FLAG_CONT_MOBILE: float
    FLAG_PHONE: float
    FLAG_EMAIL: float
    CNT_FAM_MEMBERS: float
    REGION_RATING_CLIENT: float
    REGION_RATING_CLIENT_W_CITY: float
    HOUR_APPR_PROCESS_START: float
    REG_REGION_NOT_LIVE_REGION: float
    REG_REGION_NOT_WORK_REGION: float
    LIVE_REGION_NOT_WORK_REGION: float
    REG_CITY_NOT_LIVE_CITY: float
    REG_CITY_NOT_WORK_CITY: float
    LIVE_CITY_NOT_WORK_CITY: float
    EXT_SOURCE_2: float
    OBS_30_CNT_SOCIAL_CIRCLE: float
    DEF_30_CNT_SOCIAL_CIRCLE: float
    OBS_60_CNT_SOCIAL_CIRCLE: float
    DEF_60_CNT_SOCIAL_CIRCLE: float
    DAYS_LAST_PHONE_CHANGE: float
    FLAG_DOCUMENT_2: float
    FLAG_DOCUMENT_3: float
    FLAG_DOCUMENT_4: float
    FLAG_DOCUMENT_5: float
    FLAG_DOCUMENT_6: float
    FLAG_DOCUMENT_7: float
    FLAG_DOCUMENT_8: float
    FLAG_DOCUMENT_9: float
    FLAG_DOCUMENT_10: float
    FLAG_DOCUMENT_11: float
    FLAG_DOCUMENT_12: float
    FLAG_DOCUMENT_13: float
    FLAG_DOCUMENT_14: float
    FLAG_DOCUMENT_15: float
    FLAG_DOCUMENT_16: float
    FLAG_DOCUMENT_17: float
    FLAG_DOCUMENT_18: float
    FLAG_DOCUMENT_19: float
    FLAG_DOCUMENT_20: float
    FLAG_DOCUMENT_21: float
    NAME_CONTRACT_TYPE_Cashloans: float
    NAME_CONTRACT_TYPE_Revolvingloans: float
    NAME_TYPE_SUITE_Children: float
    NAME_TYPE_SUITE_Family: float
    NAME_TYPE_SUITE_Groupofpeople: float
    NAME_TYPE_SUITE_Other_A: float
    NAME_TYPE_SUITE_Other_B: float
    NAME_TYPE_SUITE_Spousepartner: float
    NAME_TYPE_SUITE_Unaccompanied: float
    NAME_INCOME_TYPE_Businessman: float
    NAME_INCOME_TYPE_Commercialassociate: float
    NAME_INCOME_TYPE_Maternityleave: float
    NAME_INCOME_TYPE_Pensioner: float
    NAME_INCOME_TYPE_Stateservant: float
    NAME_INCOME_TYPE_Student: float
    NAME_INCOME_TYPE_Unemployed: float
    NAME_INCOME_TYPE_Working: float
    NAME_EDUCATION_TYPE_Academicdegree: float
    NAME_EDUCATION_TYPE_Highereducation: float
    NAME_EDUCATION_TYPE_Incompletehigher: float
    NAME_EDUCATION_TYPE_Lowersecondary: float
    NAME_EDUCATION_TYPE_Secondarysecondaryspecial: float
    NAME_FAMILY_STATUS_Civilmarriage: float
    NAME_FAMILY_STATUS_Married: float
    NAME_FAMILY_STATUS_Separated: float
    NAME_FAMILY_STATUS_Singlenotmarried: float
    NAME_FAMILY_STATUS_Unknown: float
    NAME_FAMILY_STATUS_Widow: float
    NAME_HOUSING_TYPE_Coopapartment: float
    NAME_HOUSING_TYPE_Houseapartment: float
    NAME_HOUSING_TYPE_Municipalapartment: float
    NAME_HOUSING_TYPE_Officeapartment: float
    NAME_HOUSING_TYPE_Rentedapartment: float
    NAME_HOUSING_TYPE_Withparents: float
    OCCUPATION_TYPE_Accountants: float
    OCCUPATION_TYPE_Cleaningstaff: float
    OCCUPATION_TYPE_Cookingstaff: float
    OCCUPATION_TYPE_Corestaff: float
    OCCUPATION_TYPE_Drivers: float
    OCCUPATION_TYPE_HRstaff: float
    OCCUPATION_TYPE_Highskilltechstaff: float
    OCCUPATION_TYPE_ITstaff: float
    OCCUPATION_TYPE_Laborers: float
    OCCUPATION_TYPE_LowskillLaborers: float
    OCCUPATION_TYPE_Managers: float
    OCCUPATION_TYPE_Medicinestaff: float
    OCCUPATION_TYPE_Privateservicestaff: float
    OCCUPATION_TYPE_Realtyagents: float
    OCCUPATION_TYPE_Salesstaff: float
    OCCUPATION_TYPE_Secretaries: float
    OCCUPATION_TYPE_Securitystaff: float
    OCCUPATION_TYPE_Waitersbarmenstaff: float
    WEEKDAY_APPR_PROCESS_START_FRIDAY: float
    WEEKDAY_APPR_PROCESS_START_MONDAY: float
    WEEKDAY_APPR_PROCESS_START_SATURDAY: float
    WEEKDAY_APPR_PROCESS_START_SUNDAY: float
    WEEKDAY_APPR_PROCESS_START_THURSDAY: float
    WEEKDAY_APPR_PROCESS_START_TUESDAY: float
    WEEKDAY_APPR_PROCESS_START_WEDNESDAY: float
    ORGANIZATION_TYPE_Advertising: float
    ORGANIZATION_TYPE_Agriculture: float
    ORGANIZATION_TYPE_Bank: float
    ORGANIZATION_TYPE_BusinessEntityType1: float
    ORGANIZATION_TYPE_BusinessEntityType2: float
    ORGANIZATION_TYPE_BusinessEntityType3: float
    ORGANIZATION_TYPE_Cleaning: float
    ORGANIZATION_TYPE_Construction: float
    ORGANIZATION_TYPE_Culture: float
    ORGANIZATION_TYPE_Electricity: float
    ORGANIZATION_TYPE_Emergency: float
    ORGANIZATION_TYPE_Government: float
    ORGANIZATION_TYPE_Hotel: float
    ORGANIZATION_TYPE_Housing: float
    ORGANIZATION_TYPE_Industrytype1: float
    ORGANIZATION_TYPE_Industrytype10: float
    ORGANIZATION_TYPE_Industrytype11: float
    ORGANIZATION_TYPE_Industrytype12: float
    ORGANIZATION_TYPE_Industrytype13: float
    ORGANIZATION_TYPE_Industrytype2: float
    ORGANIZATION_TYPE_Industrytype3: float
    ORGANIZATION_TYPE_Industrytype4: float
    ORGANIZATION_TYPE_Industrytype5: float
    ORGANIZATION_TYPE_Industrytype6: float
    ORGANIZATION_TYPE_Industrytype7: float
    ORGANIZATION_TYPE_Industrytype8: float
    ORGANIZATION_TYPE_Industrytype9: float
    ORGANIZATION_TYPE_Insurance: float
    ORGANIZATION_TYPE_Kindergarten: float
    ORGANIZATION_TYPE_LegalServices: float
    ORGANIZATION_TYPE_Medicine: float
    ORGANIZATION_TYPE_Military: float
    ORGANIZATION_TYPE_Mobile: float
    ORGANIZATION_TYPE_Other: float
    ORGANIZATION_TYPE_Police: float
    ORGANIZATION_TYPE_Postal: float
    ORGANIZATION_TYPE_Realtor: float
    ORGANIZATION_TYPE_Religion: float
    ORGANIZATION_TYPE_Restaurant: float
    ORGANIZATION_TYPE_School: float
    ORGANIZATION_TYPE_Security: float
    ORGANIZATION_TYPE_SecurityMinistries: float
    ORGANIZATION_TYPE_Selfemployed: float
    ORGANIZATION_TYPE_Services: float
    ORGANIZATION_TYPE_Telecom: float
    ORGANIZATION_TYPE_Tradetype1: float
    ORGANIZATION_TYPE_Tradetype2: float
    ORGANIZATION_TYPE_Tradetype3: float
    ORGANIZATION_TYPE_Tradetype4: float
    ORGANIZATION_TYPE_Tradetype5: float
    ORGANIZATION_TYPE_Tradetype6: float
    ORGANIZATION_TYPE_Tradetype7: float
    ORGANIZATION_TYPE_Transporttype1: float
    ORGANIZATION_TYPE_Transporttype2: float
    ORGANIZATION_TYPE_Transporttype3: float
    ORGANIZATION_TYPE_Transporttype4: float
    ORGANIZATION_TYPE_University: float
    ORGANIZATION_TYPE_XNA: float
    FONDKAPREMONT_MODE_notspecified: float
    FONDKAPREMONT_MODE_orgspecaccount: float
    FONDKAPREMONT_MODE_regoperaccount: float
    FONDKAPREMONT_MODE_regoperspecaccount: float
    HOUSETYPE_MODE_blockofflats: float
    HOUSETYPE_MODE_specifichousing: float
    HOUSETYPE_MODE_terracedhouse: float
    WALLSMATERIAL_MODE_Block: float
    WALLSMATERIAL_MODE_Mixed: float
    WALLSMATERIAL_MODE_Monolithic: float
    WALLSMATERIAL_MODE_Others: float
    WALLSMATERIAL_MODE_Panel: float
    WALLSMATERIAL_MODE_Stonebrick: float
    WALLSMATERIAL_MODE_Wooden: float
    EMERGENCYSTATE_MODE_No: float
    EMERGENCYSTATE_MODE_Yes: float
    INCOME_CREDIT_PERC: float
    INCOME_PER_PERSON: float
    ANNUITY_INCOME_PERC: float
    PAYMENT_RATE: float


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello you.'}


# 3.
@app.post('/predict')

def predict(json_data: bank_model):

    data_dict = dict(json_data)

    print(data_dict)
    print(type(json_data))

    data_df = pd.DataFrame.from_dict(data_dict, orient = 'index').T

    prediction = LGBM_model.predict([data_df.iloc[0].to_list()])
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