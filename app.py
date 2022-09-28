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

class bank_model(BaseModel):
    index: int
    SK_ID_CURR: int
    CODE_GENDER: int
    FLAG_OWN_CAR: int
    FLAG_OWN_REALTY: int
    CNT_CHILDREN: int
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: Union[float, None]
    AMT_GOODS_PRICE: Union[float, None]
    REGION_POPULATION_RELATIVE: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: Union[float, None]
    DAYS_REGISTRATION: float
    DAYS_ID_PUBLISH: int
    OWN_CAR_AGE: Union[float, None]
    FLAG_MOBIL: int
    FLAG_EMP_PHONE: int
    FLAG_WORK_PHONE: int
    FLAG_CONT_MOBILE: int
    FLAG_PHONE: int
    FLAG_EMAIL: int
    CNT_FAM_MEMBERS: Union[float, None]
    REGION_RATING_CLIENT: int
    REGION_RATING_CLIENT_W_CITY: int
    HOUR_APPR_PROCESS_START: int
    REG_REGION_NOT_LIVE_REGION: int
    REG_REGION_NOT_WORK_REGION: int
    LIVE_REGION_NOT_WORK_REGION: int
    REG_CITY_NOT_LIVE_CITY: int
    REG_CITY_NOT_WORK_CITY: int
    LIVE_CITY_NOT_WORK_CITY: int
    EXT_SOURCE_1: Union[float, None]
    EXT_SOURCE_2: Union[float, None]
    EXT_SOURCE_3: Union[float, None]
    APARTMENTS_AVG: Union[float, None]
    BASEMENTAREA_AVG: Union[float, None]
    YEARS_BEGINEXPLUATATION_AVG: Union[float, None]
    YEARS_BUILD_AVG: Union[float, None]
    COMMONAREA_AVG: Union[float, None]
    ELEVATORS_AVG: Union[float, None]
    ENTRANCES_AVG: Union[float, None]
    FLOORSMAX_AVG: Union[float, None]
    FLOORSMIN_AVG: Union[float, None]
    LANDAREA_AVG: Union[float, None]
    LIVINGAPARTMENTS_AVG: Union[float, None]
    LIVINGAREA_AVG: Union[float, None]
    NONLIVINGAPARTMENTS_AVG: Union[float, None]
    NONLIVINGAREA_AVG: Union[float, None]
    APARTMENTS_MODE: Union[float, None]
    BASEMENTAREA_MODE: Union[float, None]
    YEARS_BEGINEXPLUATATION_MODE: Union[float, None]
    YEARS_BUILD_MODE: Union[float, None]
    COMMONAREA_MODE: Union[float, None]
    ELEVATORS_MODE: Union[float, None]
    ENTRANCES_MODE: Union[float, None]
    FLOORSMAX_MODE: Union[float, None]
    FLOORSMIN_MODE: Union[float, None]
    LANDAREA_MODE: Union[float, None]
    LIVINGAPARTMENTS_MODE: Union[float, None]
    LIVINGAREA_MODE: Union[float, None]
    NONLIVINGAPARTMENTS_MODE: Union[float, None]
    NONLIVINGAREA_MODE: Union[float, None]
    APARTMENTS_MEDI: Union[float, None]
    BASEMENTAREA_MEDI: Union[float, None]
    YEARS_BEGINEXPLUATATION_MEDI: Union[float, None]
    YEARS_BUILD_MEDI: Union[float, None]
    COMMONAREA_MEDI: Union[float, None]
    ELEVATORS_MEDI: Union[float, None]
    ENTRANCES_MEDI: Union[float, None]
    FLOORSMAX_MEDI: Union[float, None]
    FLOORSMIN_MEDI: Union[float, None]
    LANDAREA_MEDI: Union[float, None]
    LIVINGAPARTMENTS_MEDI: Union[float, None]
    LIVINGAREA_MEDI: Union[float, None]
    NONLIVINGAPARTMENTS_MEDI: Union[float, None]
    NONLIVINGAREA_MEDI: Union[float, None]
    TOTALAREA_MODE: Union[float, None]
    OBS_30_CNT_SOCIAL_CIRCLE: Union[float, None]
    DEF_30_CNT_SOCIAL_CIRCLE: Union[float, None]
    OBS_60_CNT_SOCIAL_CIRCLE: Union[float, None]
    DEF_60_CNT_SOCIAL_CIRCLE: Union[float, None]
    DAYS_LAST_PHONE_CHANGE: Union[float, None]
    FLAG_DOCUMENT_2: int
    FLAG_DOCUMENT_3: int
    FLAG_DOCUMENT_4: int
    FLAG_DOCUMENT_5: int
    FLAG_DOCUMENT_6: int
    FLAG_DOCUMENT_7: int
    FLAG_DOCUMENT_8: int
    FLAG_DOCUMENT_9: int
    FLAG_DOCUMENT_10: int
    FLAG_DOCUMENT_11: int
    FLAG_DOCUMENT_12: int
    FLAG_DOCUMENT_13: int
    FLAG_DOCUMENT_14: int
    FLAG_DOCUMENT_15: int
    FLAG_DOCUMENT_16: int
    FLAG_DOCUMENT_17: int
    FLAG_DOCUMENT_18: int
    FLAG_DOCUMENT_19: int
    FLAG_DOCUMENT_20: int
    FLAG_DOCUMENT_21: int
    AMT_REQ_CREDIT_BUREAU_HOUR: Union[float, None]
    AMT_REQ_CREDIT_BUREAU_DAY: Union[float, None]
    AMT_REQ_CREDIT_BUREAU_WEEK: Union[float, None]
    AMT_REQ_CREDIT_BUREAU_MON: Union[float, None]
    AMT_REQ_CREDIT_BUREAU_QRT: Union[float, None]
    AMT_REQ_CREDIT_BUREAU_YEAR: Union[float, None]
    NAME_CONTRACT_TYPE_Cashloans: int
    NAME_CONTRACT_TYPE_Revolvingloans: int
    NAME_TYPE_SUITE_Children: int
    NAME_TYPE_SUITE_Family: int
    NAME_TYPE_SUITE_Groupofpeople: int
    NAME_TYPE_SUITE_Other_A: int
    NAME_TYPE_SUITE_Other_B: int
    NAME_TYPE_SUITE_Spousepartner: int
    NAME_TYPE_SUITE_Unaccompanied: int
    NAME_INCOME_TYPE_Businessman: int
    NAME_INCOME_TYPE_Commercialassociate: int
    NAME_INCOME_TYPE_Maternityleave: int
    NAME_INCOME_TYPE_Pensioner: int
    NAME_INCOME_TYPE_Stateservant: int
    NAME_INCOME_TYPE_Student: int
    NAME_INCOME_TYPE_Unemployed: int
    NAME_INCOME_TYPE_Working: int
    NAME_EDUCATION_TYPE_Academicdegree: int
    NAME_EDUCATION_TYPE_Highereducation: int
    NAME_EDUCATION_TYPE_Incompletehigher: int
    NAME_EDUCATION_TYPE_Lowersecondary: int
    NAME_EDUCATION_TYPE_Secondarysecondaryspecial: int
    NAME_FAMILY_STATUS_Civilmarriage: int
    NAME_FAMILY_STATUS_Married: int
    NAME_FAMILY_STATUS_Separated: int
    NAME_FAMILY_STATUS_Singlenotmarried: int
    NAME_FAMILY_STATUS_Unknown: int
    NAME_FAMILY_STATUS_Widow: int
    NAME_HOUSING_TYPE_Coopapartment: int
    NAME_HOUSING_TYPE_Houseapartment: int
    NAME_HOUSING_TYPE_Municipalapartment: int
    NAME_HOUSING_TYPE_Officeapartment: int
    NAME_HOUSING_TYPE_Rentedapartment: int
    NAME_HOUSING_TYPE_Withparents: int
    OCCUPATION_TYPE_Accountants: int
    OCCUPATION_TYPE_Cleaningstaff: int
    OCCUPATION_TYPE_Cookingstaff: int
    OCCUPATION_TYPE_Corestaff: int
    OCCUPATION_TYPE_Drivers: int
    OCCUPATION_TYPE_HRstaff: int
    OCCUPATION_TYPE_Highskilltechstaff: int
    OCCUPATION_TYPE_ITstaff: int
    OCCUPATION_TYPE_Laborers: int
    OCCUPATION_TYPE_LowskillLaborers: int
    OCCUPATION_TYPE_Managers: int
    OCCUPATION_TYPE_Medicinestaff: int
    OCCUPATION_TYPE_Privateservicestaff: int
    OCCUPATION_TYPE_Realtyagents: int
    OCCUPATION_TYPE_Salesstaff: int
    OCCUPATION_TYPE_Secretaries: int
    OCCUPATION_TYPE_Securitystaff: int
    OCCUPATION_TYPE_Waitersbarmenstaff: int
    WEEKDAY_APPR_PROCESS_START_FRIDAY: int
    WEEKDAY_APPR_PROCESS_START_MONDAY: int
    WEEKDAY_APPR_PROCESS_START_SATURDAY: int
    WEEKDAY_APPR_PROCESS_START_SUNDAY: int
    WEEKDAY_APPR_PROCESS_START_THURSDAY: int
    WEEKDAY_APPR_PROCESS_START_TUESDAY: int
    WEEKDAY_APPR_PROCESS_START_WEDNESDAY: int
    ORGANIZATION_TYPE_Advertising: int
    ORGANIZATION_TYPE_Agriculture: int
    ORGANIZATION_TYPE_Bank: int
    ORGANIZATION_TYPE_BusinessEntityType1: int
    ORGANIZATION_TYPE_BusinessEntityType2: int
    ORGANIZATION_TYPE_BusinessEntityType3: int
    ORGANIZATION_TYPE_Cleaning: int
    ORGANIZATION_TYPE_Construction: int
    ORGANIZATION_TYPE_Culture: int
    ORGANIZATION_TYPE_Electricity: int
    ORGANIZATION_TYPE_Emergency: int
    ORGANIZATION_TYPE_Government: int
    ORGANIZATION_TYPE_Hotel: int
    ORGANIZATION_TYPE_Housing: int
    ORGANIZATION_TYPE_Industrytype1: int
    ORGANIZATION_TYPE_Industrytype10: int
    ORGANIZATION_TYPE_Industrytype11: int
    ORGANIZATION_TYPE_Industrytype12: int
    ORGANIZATION_TYPE_Industrytype13: int
    ORGANIZATION_TYPE_Industrytype2: int
    ORGANIZATION_TYPE_Industrytype3: int
    ORGANIZATION_TYPE_Industrytype4: int
    ORGANIZATION_TYPE_Industrytype5: int
    ORGANIZATION_TYPE_Industrytype6: int
    ORGANIZATION_TYPE_Industrytype7: int
    ORGANIZATION_TYPE_Industrytype8: int
    ORGANIZATION_TYPE_Industrytype9: int
    ORGANIZATION_TYPE_Insurance: int
    ORGANIZATION_TYPE_Kindergarten: int
    ORGANIZATION_TYPE_LegalServices: int
    ORGANIZATION_TYPE_Medicine: int
    ORGANIZATION_TYPE_Military: int
    ORGANIZATION_TYPE_Mobile: int
    ORGANIZATION_TYPE_Other: int
    ORGANIZATION_TYPE_Police: int
    ORGANIZATION_TYPE_Postal: int
    ORGANIZATION_TYPE_Realtor: int
    ORGANIZATION_TYPE_Religion: int
    ORGANIZATION_TYPE_Restaurant: int
    ORGANIZATION_TYPE_School: int
    ORGANIZATION_TYPE_Security: int
    ORGANIZATION_TYPE_SecurityMinistries: int
    ORGANIZATION_TYPE_Selfemployed: int
    ORGANIZATION_TYPE_Services: int
    ORGANIZATION_TYPE_Telecom: int
    ORGANIZATION_TYPE_Tradetype1: int
    ORGANIZATION_TYPE_Tradetype2: int
    ORGANIZATION_TYPE_Tradetype3: int
    ORGANIZATION_TYPE_Tradetype4: int
    ORGANIZATION_TYPE_Tradetype5: int
    ORGANIZATION_TYPE_Tradetype6: int
    ORGANIZATION_TYPE_Tradetype7: int
    ORGANIZATION_TYPE_Transporttype1: int
    ORGANIZATION_TYPE_Transporttype2: int
    ORGANIZATION_TYPE_Transporttype3: int
    ORGANIZATION_TYPE_Transporttype4: int
    ORGANIZATION_TYPE_University: int
    ORGANIZATION_TYPE_XNA: int
    FONDKAPREMONT_MODE_notspecified: int
    FONDKAPREMONT_MODE_orgspecaccount: int
    FONDKAPREMONT_MODE_regoperaccount: int
    FONDKAPREMONT_MODE_regoperspecaccount: int
    HOUSETYPE_MODE_blockofflats: int
    HOUSETYPE_MODE_specifichousing: int
    HOUSETYPE_MODE_terracedhouse: int
    WALLSMATERIAL_MODE_Block: int
    WALLSMATERIAL_MODE_Mixed: int
    WALLSMATERIAL_MODE_Monolithic: int
    WALLSMATERIAL_MODE_Others: int
    WALLSMATERIAL_MODE_Panel: int
    WALLSMATERIAL_MODE_Stonebrick: int
    WALLSMATERIAL_MODE_Wooden: int
    EMERGENCYSTATE_MODE_No: int
    EMERGENCYSTATE_MODE_Yes: int
    DAYS_EMPLOYED_PERC: Union[float, None]
    INCOME_CREDIT_PERC: float
    INCOME_PER_PERSON: Union[float, None]
    ANNUITY_INCOME_PERC: Union[float, None]
    PAYMENT_RATE: Union[float, None]
    BURO_DAYS_CREDIT_MIN: Union[float, None]
    BURO_DAYS_CREDIT_MAX: Union[float, None]
    BURO_DAYS_CREDIT_MEAN: Union[float, None]
    BURO_DAYS_CREDIT_VAR: Union[float, None]
    BURO_DAYS_CREDIT_ENDDATE_MIN: Union[float, None]
    BURO_DAYS_CREDIT_ENDDATE_MAX: Union[float, None]
    BURO_DAYS_CREDIT_ENDDATE_MEAN: Union[float, None]
    BURO_DAYS_CREDIT_UPDATE_MEAN: Union[float, None]
    BURO_CREDIT_DAY_OVERDUE_MAX: Union[float, None]
    BURO_CREDIT_DAY_OVERDUE_MEAN: Union[float, None]
    BURO_AMT_CREDIT_MAX_OVERDUE_MEAN: Union[float, None]
    BURO_AMT_CREDIT_SUM_MAX: Union[float, None]
    BURO_AMT_CREDIT_SUM_MEAN: Union[float, None]
    BURO_AMT_CREDIT_SUM_SUM: Union[float, None]
    BURO_AMT_CREDIT_SUM_DEBT_MAX: Union[float, None]
    BURO_AMT_CREDIT_SUM_DEBT_MEAN: Union[float, None]
    BURO_AMT_CREDIT_SUM_DEBT_SUM: Union[float, None]
    BURO_AMT_CREDIT_SUM_OVERDUE_MEAN: Union[float, None]
    BURO_AMT_CREDIT_SUM_LIMIT_MEAN: Union[float, None]
    BURO_AMT_CREDIT_SUM_LIMIT_SUM: Union[float, None]
    BURO_AMT_ANNUITY_MAX: Union[float, None]
    BURO_AMT_ANNUITY_MEAN: Union[float, None]
    BURO_CNT_CREDIT_PROLONG_SUM: Union[float, None]
    BURO_MONTHS_BALANCE_MIN_MIN: Union[float, None]
    BURO_MONTHS_BALANCE_MAX_MAX: Union[float, None]
    BURO_MONTHS_BALANCE_SIZE_MEAN: Union[float, None]
    BURO_MONTHS_BALANCE_SIZE_SUM: Union[float, None]
    BURO_CREDIT_ACTIVE_Active_MEAN: Union[float, None]
    BURO_CREDIT_ACTIVE_Baddebt_MEAN: Union[float, None]
    BURO_CREDIT_ACTIVE_Closed_MEAN: Union[float, None]
    BURO_CREDIT_ACTIVE_Sold_MEAN: Union[float, None]
    BURO_CREDIT_ACTIVE_nan_MEAN: Union[float, None]
    BURO_CREDIT_CURRENCY_currency1_MEAN: Union[float, None]
    BURO_CREDIT_CURRENCY_currency2_MEAN: Union[float, None]
    BURO_CREDIT_CURRENCY_currency3_MEAN: Union[float, None]
    BURO_CREDIT_CURRENCY_currency4_MEAN: Union[float, None]
    BURO_CREDIT_CURRENCY_nan_MEAN: Union[float, None]
    BURO_CREDIT_TYPE_Anothertypeofloan_MEAN: Union[float, None]
    BURO_CREDIT_TYPE_Carloan_MEAN: Union[float, None]
    BURO_CREDIT_TYPE_Cashloannonearmarked_MEAN: Union[float, None]
    BURO_CREDIT_TYPE_Consumercredit_MEAN: Union[float, None]
    BURO_CREDIT_TYPE_Creditcard_MEAN: Union[float, None]
    BURO_CREDIT_TYPE_Interbankcredit_MEAN: Union[float, None]
    BURO_CREDIT_TYPE_Loanforbusinessdevelopment_MEAN: Union[float, None]
    BURO_CREDIT_TYPE_Loanforpurchaseofsharesmarginlending_MEAN: Union[float, None]
    BURO_CREDIT_TYPE_Loanforthepurchaseofequipment_MEAN: Union[float, None]
    BURO_CREDIT_TYPE_Loanforworkingcapitalreplenishment_MEAN: Union[float, None]
    BURO_CREDIT_TYPE_Microloan_MEAN: Union[float, None]
    BURO_CREDIT_TYPE_Mobileoperatorloan_MEAN: Union[float, None]
    BURO_CREDIT_TYPE_Mortgage_MEAN: Union[float, None]
    BURO_CREDIT_TYPE_Realestateloan_MEAN: Union[float, None]
    BURO_CREDIT_TYPE_Unknowntypeofloan_MEAN: Union[float, None]
    BURO_CREDIT_TYPE_nan_MEAN: Union[float, None]
    BURO_STATUS_0_MEAN_MEAN: Union[float, None]
    BURO_STATUS_1_MEAN_MEAN: Union[float, None]
    BURO_STATUS_2_MEAN_MEAN: Union[float, None]
    BURO_STATUS_3_MEAN_MEAN: Union[float, None]
    BURO_STATUS_4_MEAN_MEAN: Union[float, None]
    BURO_STATUS_5_MEAN_MEAN: Union[float, None]
    BURO_STATUS_C_MEAN_MEAN: Union[float, None]
    BURO_STATUS_X_MEAN_MEAN: Union[float, None]
    BURO_STATUS_nan_MEAN_MEAN: Union[float, None]
    ACTIVE_DAYS_CREDIT_MIN: Union[float, None]
    ACTIVE_DAYS_CREDIT_MAX: Union[float, None]
    ACTIVE_DAYS_CREDIT_MEAN: Union[float, None]
    ACTIVE_DAYS_CREDIT_VAR: Union[float, None]
    ACTIVE_DAYS_CREDIT_ENDDATE_MIN: Union[float, None]
    ACTIVE_DAYS_CREDIT_ENDDATE_MAX: Union[float, None]
    ACTIVE_DAYS_CREDIT_ENDDATE_MEAN: Union[float, None]
    ACTIVE_DAYS_CREDIT_UPDATE_MEAN: Union[float, None]
    ACTIVE_CREDIT_DAY_OVERDUE_MAX: Union[float, None]
    ACTIVE_CREDIT_DAY_OVERDUE_MEAN: Union[float, None]
    ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN: Union[float, None]
    ACTIVE_AMT_CREDIT_SUM_MAX: Union[float, None]
    ACTIVE_AMT_CREDIT_SUM_MEAN: Union[float, None]
    ACTIVE_AMT_CREDIT_SUM_SUM: Union[float, None]
    ACTIVE_AMT_CREDIT_SUM_DEBT_MAX: Union[float, None]
    ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN: Union[float, None]
    ACTIVE_AMT_CREDIT_SUM_DEBT_SUM: Union[float, None]
    ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN: Union[float, None]
    ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN: Union[float, None]
    ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM: Union[float, None]
    ACTIVE_AMT_ANNUITY_MAX: Union[float, None]
    ACTIVE_AMT_ANNUITY_MEAN: Union[float, None]
    ACTIVE_CNT_CREDIT_PROLONG_SUM: Union[float, None]
    ACTIVE_MONTHS_BALANCE_MIN_MIN: Union[float, None]
    ACTIVE_MONTHS_BALANCE_MAX_MAX: Union[float, None]
    ACTIVE_MONTHS_BALANCE_SIZE_MEAN: Union[float, None]
    ACTIVE_MONTHS_BALANCE_SIZE_SUM: Union[float, None]
    CLOSED_DAYS_CREDIT_MIN: Union[float, None]
    CLOSED_DAYS_CREDIT_MAX: Union[float, None]
    CLOSED_DAYS_CREDIT_MEAN: Union[float, None]
    CLOSED_DAYS_CREDIT_VAR: Union[float, None]
    CLOSED_DAYS_CREDIT_ENDDATE_MIN: Union[float, None]
    CLOSED_DAYS_CREDIT_ENDDATE_MAX: Union[float, None]
    CLOSED_DAYS_CREDIT_ENDDATE_MEAN: Union[float, None]
    CLOSED_DAYS_CREDIT_UPDATE_MEAN: Union[float, None]
    CLOSED_CREDIT_DAY_OVERDUE_MAX: Union[float, None]
    CLOSED_CREDIT_DAY_OVERDUE_MEAN: Union[float, None]
    CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN: Union[float, None]
    CLOSED_AMT_CREDIT_SUM_MAX: Union[float, None]
    CLOSED_AMT_CREDIT_SUM_MEAN: Union[float, None]
    CLOSED_AMT_CREDIT_SUM_SUM: Union[float, None]
    CLOSED_AMT_CREDIT_SUM_DEBT_MAX: Union[float, None]
    CLOSED_AMT_CREDIT_SUM_DEBT_MEAN: Union[float, None]
    CLOSED_AMT_CREDIT_SUM_DEBT_SUM: Union[float, None]
    CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN: Union[float, None]
    CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN: Union[float, None]
    CLOSED_AMT_CREDIT_SUM_LIMIT_SUM: Union[float, None]
    CLOSED_AMT_ANNUITY_MAX: Union[float, None]
    CLOSED_AMT_ANNUITY_MEAN: Union[float, None]
    CLOSED_CNT_CREDIT_PROLONG_SUM: Union[float, None]
    CLOSED_MONTHS_BALANCE_MIN_MIN: Union[float, None]
    CLOSED_MONTHS_BALANCE_MAX_MAX: Union[float, None]
    CLOSED_MONTHS_BALANCE_SIZE_MEAN: Union[float, None]
    CLOSED_MONTHS_BALANCE_SIZE_SUM: Union[float, None]
    PREV_AMT_ANNUITY_MIN: Union[float, None]
    PREV_AMT_ANNUITY_MAX: Union[float, None]
    PREV_AMT_ANNUITY_MEAN: Union[float, None]
    PREV_AMT_APPLICATION_MIN: Union[float, None]
    PREV_AMT_APPLICATION_MAX: Union[float, None]
    PREV_AMT_APPLICATION_MEAN: Union[float, None]
    PREV_AMT_CREDIT_MIN: Union[float, None]
    PREV_AMT_CREDIT_MAX: Union[float, None]
    PREV_AMT_CREDIT_MEAN: Union[float, None]
    PREV_APP_CREDIT_PERC_MIN: Union[float, None]
    PREV_APP_CREDIT_PERC_MAX: Union[float, None]
    PREV_APP_CREDIT_PERC_MEAN: Union[float, None]
    PREV_APP_CREDIT_PERC_VAR: Union[float, None]
    PREV_AMT_DOWN_PAYMENT_MIN: Union[float, None]
    PREV_AMT_DOWN_PAYMENT_MAX: Union[float, None]
    PREV_AMT_DOWN_PAYMENT_MEAN: Union[float, None]
    PREV_AMT_GOODS_PRICE_MIN: Union[float, None]
    PREV_AMT_GOODS_PRICE_MAX: Union[float, None]
    PREV_AMT_GOODS_PRICE_MEAN: Union[float, None]
    PREV_HOUR_APPR_PROCESS_START_MIN: Union[float, None]
    PREV_HOUR_APPR_PROCESS_START_MAX: Union[float, None]
    PREV_HOUR_APPR_PROCESS_START_MEAN: Union[float, None]
    PREV_RATE_DOWN_PAYMENT_MIN: Union[float, None]
    PREV_RATE_DOWN_PAYMENT_MAX: Union[float, None]
    PREV_RATE_DOWN_PAYMENT_MEAN: Union[float, None]
    PREV_DAYS_DECISION_MIN: Union[float, None]
    PREV_DAYS_DECISION_MAX: Union[float, None]
    PREV_DAYS_DECISION_MEAN: Union[float, None]
    PREV_CNT_PAYMENT_MEAN: Union[float, None]
    PREV_CNT_PAYMENT_SUM: Union[float, None]
    PREV_NAME_CONTRACT_TYPE_Cashloans_MEAN: Union[float, None]
    PREV_NAME_CONTRACT_TYPE_Consumerloans_MEAN: Union[float, None]
    PREV_NAME_CONTRACT_TYPE_Revolvingloans_MEAN: Union[float, None]
    PREV_NAME_CONTRACT_TYPE_XNA_MEAN: Union[float, None]
    PREV_NAME_CONTRACT_TYPE_nan_MEAN: Union[float, None]
    PREV_WEEKDAY_APPR_PROCESS_START_FRIDAY_MEAN: Union[float, None]
    PREV_WEEKDAY_APPR_PROCESS_START_MONDAY_MEAN: Union[float, None]
    PREV_WEEKDAY_APPR_PROCESS_START_SATURDAY_MEAN: Union[float, None]
    PREV_WEEKDAY_APPR_PROCESS_START_SUNDAY_MEAN: Union[float, None]
    PREV_WEEKDAY_APPR_PROCESS_START_THURSDAY_MEAN: Union[float, None]
    PREV_WEEKDAY_APPR_PROCESS_START_TUESDAY_MEAN: Union[float, None]
    PREV_WEEKDAY_APPR_PROCESS_START_WEDNESDAY_MEAN: Union[float, None]
    PREV_WEEKDAY_APPR_PROCESS_START_nan_MEAN: Union[float, None]
    PREV_FLAG_LAST_APPL_PER_CONTRACT_N_MEAN: Union[float, None]
    PREV_FLAG_LAST_APPL_PER_CONTRACT_Y_MEAN: Union[float, None]
    PREV_FLAG_LAST_APPL_PER_CONTRACT_nan_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_Buildingahouseoranannex_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_Businessdevelopment_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_Buyingagarage_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_Buyingaholidayhomeland_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_Buyingahome_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_Buyinganewcar_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_Buyingausedcar_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_Carrepairs_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_Education_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_Everydayexpenses_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_Furniture_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_Gasificationwatersupply_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_Hobby_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_Journey_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_Medicine_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_Moneyforathirdperson_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_Other_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_Paymentsonotherloans_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_Purchaseofelectronicequipment_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_Refusaltonamethegoal_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_Repairs_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_Urgentneeds_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_Weddinggiftholiday_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_XAP_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_XNA_MEAN: Union[float, None]
    PREV_NAME_CASH_LOAN_PURPOSE_nan_MEAN: Union[float, None]
    PREV_NAME_CONTRACT_STATUS_Approved_MEAN: Union[float, None]
    PREV_NAME_CONTRACT_STATUS_Canceled_MEAN: Union[float, None]
    PREV_NAME_CONTRACT_STATUS_Refused_MEAN: Union[float, None]
    PREV_NAME_CONTRACT_STATUS_Unusedoffer_MEAN: Union[float, None]
    PREV_NAME_CONTRACT_STATUS_nan_MEAN: Union[float, None]
    PREV_NAME_PAYMENT_TYPE_Cashthroughthebank_MEAN: Union[float, None]
    PREV_NAME_PAYMENT_TYPE_Cashlessfromtheaccountoftheemployer_MEAN: Union[float, None]
    PREV_NAME_PAYMENT_TYPE_Noncashfromyouraccount_MEAN: Union[float, None]
    PREV_NAME_PAYMENT_TYPE_XNA_MEAN: Union[float, None]
    PREV_NAME_PAYMENT_TYPE_nan_MEAN: Union[float, None]
    PREV_CODE_REJECT_REASON_CLIENT_MEAN: Union[float, None]
    PREV_CODE_REJECT_REASON_HC_MEAN: Union[float, None]
    PREV_CODE_REJECT_REASON_LIMIT_MEAN: Union[float, None]
    PREV_CODE_REJECT_REASON_SCO_MEAN: Union[float, None]
    PREV_CODE_REJECT_REASON_SCOFR_MEAN: Union[float, None]
    PREV_CODE_REJECT_REASON_SYSTEM_MEAN: Union[float, None]
    PREV_CODE_REJECT_REASON_VERIF_MEAN: Union[float, None]
    PREV_CODE_REJECT_REASON_XAP_MEAN: Union[float, None]
    PREV_CODE_REJECT_REASON_XNA_MEAN: Union[float, None]
    PREV_CODE_REJECT_REASON_nan_MEAN: Union[float, None]
    PREV_NAME_TYPE_SUITE_Children_MEAN: Union[float, None]
    PREV_NAME_TYPE_SUITE_Family_MEAN: Union[float, None]
    PREV_NAME_TYPE_SUITE_Groupofpeople_MEAN: Union[float, None]
    PREV_NAME_TYPE_SUITE_Other_A_MEAN: Union[float, None]
    PREV_NAME_TYPE_SUITE_Other_B_MEAN: Union[float, None]
    PREV_NAME_TYPE_SUITE_Spousepartner_MEAN: Union[float, None]
    PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN: Union[float, None]
    PREV_NAME_TYPE_SUITE_nan_MEAN: Union[float, None]
    PREV_NAME_CLIENT_TYPE_New_MEAN: Union[float, None]
    PREV_NAME_CLIENT_TYPE_Refreshed_MEAN: Union[float, None]
    PREV_NAME_CLIENT_TYPE_Repeater_MEAN: Union[float, None]
    PREV_NAME_CLIENT_TYPE_XNA_MEAN: Union[float, None]
    PREV_NAME_CLIENT_TYPE_nan_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_AdditionalService_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_Animals_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_AudioVideo_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_AutoAccessories_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_ClothingandAccessories_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_Computers_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_ConstructionMaterials_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_ConsumerElectronics_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_DirectSales_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_Education_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_Fitness_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_Furniture_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_Gardening_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_Homewares_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_HouseConstruction_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_Insurance_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_Jewelry_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_MedicalSupplies_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_Medicine_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_Mobile_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_OfficeAppliances_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_Other_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_PhotoCinemaEquipment_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_SportandLeisure_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_Tourism_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_Vehicles_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_Weapon_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_XNA_MEAN: Union[float, None]
    PREV_NAME_GOODS_CATEGORY_nan_MEAN: Union[float, None]
    PREV_NAME_PORTFOLIO_Cards_MEAN: Union[float, None]
    PREV_NAME_PORTFOLIO_Cars_MEAN: Union[float, None]
    PREV_NAME_PORTFOLIO_Cash_MEAN: Union[float, None]
    PREV_NAME_PORTFOLIO_POS_MEAN: Union[float, None]
    PREV_NAME_PORTFOLIO_XNA_MEAN: Union[float, None]
    PREV_NAME_PORTFOLIO_nan_MEAN: Union[float, None]
    PREV_NAME_PRODUCT_TYPE_XNA_MEAN: Union[float, None]
    PREV_NAME_PRODUCT_TYPE_walkin_MEAN: Union[float, None]
    PREV_NAME_PRODUCT_TYPE_xsell_MEAN: Union[float, None]
    PREV_NAME_PRODUCT_TYPE_nan_MEAN: Union[float, None]
    PREV_CHANNEL_TYPE_APCashloan_MEAN: Union[float, None]
    PREV_CHANNEL_TYPE_Cardealer_MEAN: Union[float, None]
    PREV_CHANNEL_TYPE_Channelofcorporatesales_MEAN: Union[float, None]
    PREV_CHANNEL_TYPE_Contactcenter_MEAN: Union[float, None]
    PREV_CHANNEL_TYPE_Countrywide_MEAN: Union[float, None]
    PREV_CHANNEL_TYPE_Creditandcashoffices_MEAN: Union[float, None]
    PREV_CHANNEL_TYPE_RegionalLocal_MEAN: Union[float, None]
    PREV_CHANNEL_TYPE_Stone_MEAN: Union[float, None]
    PREV_CHANNEL_TYPE_nan_MEAN: Union[float, None]
    PREV_NAME_SELLER_INDUSTRY_Autotechnology_MEAN: Union[float, None]
    PREV_NAME_SELLER_INDUSTRY_Clothing_MEAN: Union[float, None]
    PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN: Union[float, None]
    PREV_NAME_SELLER_INDUSTRY_Construction_MEAN: Union[float, None]
    PREV_NAME_SELLER_INDUSTRY_Consumerelectronics_MEAN: Union[float, None]
    PREV_NAME_SELLER_INDUSTRY_Furniture_MEAN: Union[float, None]
    PREV_NAME_SELLER_INDUSTRY_Industry_MEAN: Union[float, None]
    PREV_NAME_SELLER_INDUSTRY_Jewelry_MEAN: Union[float, None]
    PREV_NAME_SELLER_INDUSTRY_MLMpartners_MEAN: Union[float, None]
    PREV_NAME_SELLER_INDUSTRY_Tourism_MEAN: Union[float, None]
    PREV_NAME_SELLER_INDUSTRY_XNA_MEAN: Union[float, None]
    PREV_NAME_SELLER_INDUSTRY_nan_MEAN: Union[float, None]
    PREV_NAME_YIELD_GROUP_XNA_MEAN: Union[float, None]
    PREV_NAME_YIELD_GROUP_high_MEAN: Union[float, None]
    PREV_NAME_YIELD_GROUP_low_action_MEAN: Union[float, None]
    PREV_NAME_YIELD_GROUP_low_normal_MEAN: Union[float, None]
    PREV_NAME_YIELD_GROUP_middle_MEAN: Union[float, None]
    PREV_NAME_YIELD_GROUP_nan_MEAN: Union[float, None]
    PREV_PRODUCT_COMBINATION_CardStreet_MEAN: Union[float, None]
    PREV_PRODUCT_COMBINATION_CardXSell_MEAN: Union[float, None]
    PREV_PRODUCT_COMBINATION_Cash_MEAN: Union[float, None]
    PREV_PRODUCT_COMBINATION_CashStreethigh_MEAN: Union[float, None]
    PREV_PRODUCT_COMBINATION_CashStreetlow_MEAN: Union[float, None]
    PREV_PRODUCT_COMBINATION_CashStreetmiddle_MEAN: Union[float, None]
    PREV_PRODUCT_COMBINATION_CashXSellhigh_MEAN: Union[float, None]
    PREV_PRODUCT_COMBINATION_CashXSelllow_MEAN: Union[float, None]
    PREV_PRODUCT_COMBINATION_CashXSellmiddle_MEAN: Union[float, None]
    PREV_PRODUCT_COMBINATION_POShouseholdwithinterest_MEAN: Union[float, None]
    PREV_PRODUCT_COMBINATION_POShouseholdwithoutinterest_MEAN: Union[float, None]
    PREV_PRODUCT_COMBINATION_POSindustrywithinterest_MEAN: Union[float, None]
    PREV_PRODUCT_COMBINATION_POSindustrywithoutinterest_MEAN: Union[float, None]
    PREV_PRODUCT_COMBINATION_POSmobilewithinterest_MEAN: Union[float, None]
    PREV_PRODUCT_COMBINATION_POSmobilewithoutinterest_MEAN: Union[float, None]
    PREV_PRODUCT_COMBINATION_POSotherwithinterest_MEAN: Union[float, None]
    PREV_PRODUCT_COMBINATION_POSotherswithoutinterest_MEAN: Union[float, None]
    PREV_PRODUCT_COMBINATION_nan_MEAN: Union[float, None]
    APPROVED_AMT_ANNUITY_MIN: Union[float, None]
    APPROVED_AMT_ANNUITY_MAX: Union[float, None]
    APPROVED_AMT_ANNUITY_MEAN: Union[float, None]
    APPROVED_AMT_APPLICATION_MIN: Union[float, None]
    APPROVED_AMT_APPLICATION_MAX: Union[float, None]
    APPROVED_AMT_APPLICATION_MEAN: Union[float, None]
    APPROVED_AMT_CREDIT_MIN: Union[float, None]
    APPROVED_AMT_CREDIT_MAX: Union[float, None]
    APPROVED_AMT_CREDIT_MEAN: Union[float, None]
    APPROVED_APP_CREDIT_PERC_MIN: Union[float, None]
    APPROVED_APP_CREDIT_PERC_MAX: Union[float, None]
    APPROVED_APP_CREDIT_PERC_MEAN: Union[float, None]
    APPROVED_APP_CREDIT_PERC_VAR: Union[float, None]
    APPROVED_AMT_DOWN_PAYMENT_MIN: Union[float, None]
    APPROVED_AMT_DOWN_PAYMENT_MAX: Union[float, None]
    APPROVED_AMT_DOWN_PAYMENT_MEAN: Union[float, None]
    APPROVED_AMT_GOODS_PRICE_MIN: Union[float, None]
    APPROVED_AMT_GOODS_PRICE_MAX: Union[float, None]
    APPROVED_AMT_GOODS_PRICE_MEAN: Union[float, None]
    APPROVED_HOUR_APPR_PROCESS_START_MIN: Union[float, None]
    APPROVED_HOUR_APPR_PROCESS_START_MAX: Union[float, None]
    APPROVED_HOUR_APPR_PROCESS_START_MEAN: Union[float, None]
    APPROVED_RATE_DOWN_PAYMENT_MIN: Union[float, None]
    APPROVED_RATE_DOWN_PAYMENT_MAX: Union[float, None]
    APPROVED_RATE_DOWN_PAYMENT_MEAN: Union[float, None]
    APPROVED_DAYS_DECISION_MIN: Union[float, None]
    APPROVED_DAYS_DECISION_MAX: Union[float, None]
    APPROVED_DAYS_DECISION_MEAN: Union[float, None]
    APPROVED_CNT_PAYMENT_MEAN: Union[float, None]
    APPROVED_CNT_PAYMENT_SUM: Union[float, None]
    REFUSED_AMT_ANNUITY_MIN: Union[float, None]
    REFUSED_AMT_ANNUITY_MAX: Union[float, None]
    REFUSED_AMT_ANNUITY_MEAN: Union[float, None]
    REFUSED_AMT_APPLICATION_MIN: Union[float, None]
    REFUSED_AMT_APPLICATION_MAX: Union[float, None]
    REFUSED_AMT_APPLICATION_MEAN: Union[float, None]
    REFUSED_AMT_CREDIT_MIN: Union[float, None]
    REFUSED_AMT_CREDIT_MAX: Union[float, None]
    REFUSED_AMT_CREDIT_MEAN: Union[float, None]
    REFUSED_APP_CREDIT_PERC_MIN: Union[float, None]
    REFUSED_APP_CREDIT_PERC_MAX: Union[float, None]
    REFUSED_APP_CREDIT_PERC_MEAN: Union[float, None]
    REFUSED_APP_CREDIT_PERC_VAR: Union[float, None]
    REFUSED_AMT_DOWN_PAYMENT_MIN: Union[float, None]
    REFUSED_AMT_DOWN_PAYMENT_MAX: Union[float, None]
    REFUSED_AMT_DOWN_PAYMENT_MEAN: Union[float, None]
    REFUSED_AMT_GOODS_PRICE_MIN: Union[float, None]
    REFUSED_AMT_GOODS_PRICE_MAX: Union[float, None]
    REFUSED_AMT_GOODS_PRICE_MEAN: Union[float, None]
    REFUSED_HOUR_APPR_PROCESS_START_MIN: Union[float, None]
    REFUSED_HOUR_APPR_PROCESS_START_MAX: Union[float, None]
    REFUSED_HOUR_APPR_PROCESS_START_MEAN: Union[float, None]
    REFUSED_RATE_DOWN_PAYMENT_MIN: Union[float, None]
    REFUSED_RATE_DOWN_PAYMENT_MAX: Union[float, None]
    REFUSED_RATE_DOWN_PAYMENT_MEAN: Union[float, None]
    REFUSED_DAYS_DECISION_MIN: Union[float, None]
    REFUSED_DAYS_DECISION_MAX: Union[float, None]
    REFUSED_DAYS_DECISION_MEAN: Union[float, None]
    REFUSED_CNT_PAYMENT_MEAN: Union[float, None]
    REFUSED_CNT_PAYMENT_SUM: Union[float, None]
    POS_MONTHS_BALANCE_MAX: Union[float, None]
    POS_MONTHS_BALANCE_MEAN: Union[float, None]
    POS_MONTHS_BALANCE_SIZE: Union[float, None]
    POS_SK_DPD_MAX: Union[float, None]
    POS_SK_DPD_MEAN: Union[float, None]
    POS_SK_DPD_DEF_MAX: Union[float, None]
    POS_SK_DPD_DEF_MEAN: Union[float, None]
    POS_NAME_CONTRACT_STATUS_Active_MEAN: Union[float, None]
    POS_NAME_CONTRACT_STATUS_Amortizeddebt_MEAN: Union[float, None]
    POS_NAME_CONTRACT_STATUS_Approved_MEAN: Union[float, None]
    POS_NAME_CONTRACT_STATUS_Canceled_MEAN: Union[float, None]
    POS_NAME_CONTRACT_STATUS_Completed_MEAN: Union[float, None]
    POS_NAME_CONTRACT_STATUS_Demand_MEAN: Union[float, None]
    POS_NAME_CONTRACT_STATUS_Returnedtothestore_MEAN: Union[float, None]
    POS_NAME_CONTRACT_STATUS_Signed_MEAN: Union[float, None]
    POS_NAME_CONTRACT_STATUS_XNA_MEAN: Union[float, None]
    POS_NAME_CONTRACT_STATUS_nan_MEAN: Union[float, None]
    POS_COUNT: Union[float, None]
    INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE: Union[float, None]
    INSTAL_DPD_MAX: Union[float, None]
    INSTAL_DPD_MEAN: Union[float, None]
    INSTAL_DPD_SUM: Union[float, None]
    INSTAL_DBD_MAX: Union[float, None]
    INSTAL_DBD_MEAN: Union[float, None]
    INSTAL_DBD_SUM: Union[float, None]
    INSTAL_PAYMENT_PERC_MAX: Union[float, None]
    INSTAL_PAYMENT_PERC_MEAN: Union[float, None]
    INSTAL_PAYMENT_PERC_SUM: Union[float, None]
    INSTAL_PAYMENT_PERC_VAR: Union[float, None]
    INSTAL_PAYMENT_DIFF_MAX: Union[float, None]
    INSTAL_PAYMENT_DIFF_MEAN: Union[float, None]
    INSTAL_PAYMENT_DIFF_SUM: Union[float, None]
    INSTAL_PAYMENT_DIFF_VAR: Union[float, None]
    INSTAL_AMT_INSTALMENT_MAX: Union[float, None]
    INSTAL_AMT_INSTALMENT_MEAN: Union[float, None]
    INSTAL_AMT_INSTALMENT_SUM: Union[float, None]
    INSTAL_AMT_PAYMENT_MIN: Union[float, None]
    INSTAL_AMT_PAYMENT_MAX: Union[float, None]
    INSTAL_AMT_PAYMENT_MEAN: Union[float, None]
    INSTAL_AMT_PAYMENT_SUM: Union[float, None]
    INSTAL_DAYS_ENTRY_PAYMENT_MAX: Union[float, None]
    INSTAL_DAYS_ENTRY_PAYMENT_MEAN: Union[float, None]
    INSTAL_DAYS_ENTRY_PAYMENT_SUM: Union[float, None]
    INSTAL_COUNT: Union[float, None]
    CC_MONTHS_BALANCE_MIN: Union[float, None]
    CC_MONTHS_BALANCE_MAX: Union[float, None]
    CC_MONTHS_BALANCE_MEAN: Union[float, None]
    CC_MONTHS_BALANCE_SUM: Union[float, None]
    CC_MONTHS_BALANCE_VAR: Union[float, None]
    CC_AMT_BALANCE_MIN: Union[float, None]
    CC_AMT_BALANCE_MAX: Union[float, None]
    CC_AMT_BALANCE_MEAN: Union[float, None]
    CC_AMT_BALANCE_SUM: Union[float, None]
    CC_AMT_BALANCE_VAR: Union[float, None]
    CC_AMT_CREDIT_LIMIT_ACTUAL_MIN: Union[float, None]
    CC_AMT_CREDIT_LIMIT_ACTUAL_MAX: Union[float, None]
    CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN: Union[float, None]
    CC_AMT_CREDIT_LIMIT_ACTUAL_SUM: Union[float, None]
    CC_AMT_CREDIT_LIMIT_ACTUAL_VAR: Union[float, None]
    CC_AMT_DRAWINGS_ATM_CURRENT_MIN: Union[float, None]
    CC_AMT_DRAWINGS_ATM_CURRENT_MAX: Union[float, None]
    CC_AMT_DRAWINGS_ATM_CURRENT_MEAN: Union[float, None]
    CC_AMT_DRAWINGS_ATM_CURRENT_SUM: Union[float, None]
    CC_AMT_DRAWINGS_ATM_CURRENT_VAR: Union[float, None]
    CC_AMT_DRAWINGS_CURRENT_MIN: Union[float, None]
    CC_AMT_DRAWINGS_CURRENT_MAX: Union[float, None]
    CC_AMT_DRAWINGS_CURRENT_MEAN: Union[float, None]
    CC_AMT_DRAWINGS_CURRENT_SUM: Union[float, None]
    CC_AMT_DRAWINGS_CURRENT_VAR: Union[float, None]
    CC_AMT_DRAWINGS_OTHER_CURRENT_MIN: Union[float, None]
    CC_AMT_DRAWINGS_OTHER_CURRENT_MAX: Union[float, None]
    CC_AMT_DRAWINGS_OTHER_CURRENT_MEAN: Union[float, None]
    CC_AMT_DRAWINGS_OTHER_CURRENT_SUM: Union[float, None]
    CC_AMT_DRAWINGS_OTHER_CURRENT_VAR: Union[float, None]
    CC_AMT_DRAWINGS_POS_CURRENT_MIN: Union[float, None]
    CC_AMT_DRAWINGS_POS_CURRENT_MAX: Union[float, None]
    CC_AMT_DRAWINGS_POS_CURRENT_MEAN: Union[float, None]
    CC_AMT_DRAWINGS_POS_CURRENT_SUM: Union[float, None]
    CC_AMT_DRAWINGS_POS_CURRENT_VAR: Union[float, None]
    CC_AMT_INST_MIN_REGULARITY_MIN: Union[float, None]
    CC_AMT_INST_MIN_REGULARITY_MAX: Union[float, None]
    CC_AMT_INST_MIN_REGULARITY_MEAN: Union[float, None]
    CC_AMT_INST_MIN_REGULARITY_SUM: Union[float, None]
    CC_AMT_INST_MIN_REGULARITY_VAR: Union[float, None]
    CC_AMT_PAYMENT_CURRENT_MIN: Union[float, None]
    CC_AMT_PAYMENT_CURRENT_MAX: Union[float, None]
    CC_AMT_PAYMENT_CURRENT_MEAN: Union[float, None]
    CC_AMT_PAYMENT_CURRENT_SUM: Union[float, None]
    CC_AMT_PAYMENT_CURRENT_VAR: Union[float, None]
    CC_AMT_PAYMENT_TOTAL_CURRENT_MIN: Union[float, None]
    CC_AMT_PAYMENT_TOTAL_CURRENT_MAX: Union[float, None]
    CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN: Union[float, None]
    CC_AMT_PAYMENT_TOTAL_CURRENT_SUM: Union[float, None]
    CC_AMT_PAYMENT_TOTAL_CURRENT_VAR: Union[float, None]
    CC_AMT_RECEIVABLE_PRINCIPAL_MIN: Union[float, None]
    CC_AMT_RECEIVABLE_PRINCIPAL_MAX: Union[float, None]
    CC_AMT_RECEIVABLE_PRINCIPAL_MEAN: Union[float, None]
    CC_AMT_RECEIVABLE_PRINCIPAL_SUM: Union[float, None]
    CC_AMT_RECEIVABLE_PRINCIPAL_VAR: Union[float, None]
    CC_AMT_RECIVABLE_MIN: Union[float, None]
    CC_AMT_RECIVABLE_MAX: Union[float, None]
    CC_AMT_RECIVABLE_MEAN: Union[float, None]
    CC_AMT_RECIVABLE_SUM: Union[float, None]
    CC_AMT_RECIVABLE_VAR: Union[float, None]
    CC_AMT_TOTAL_RECEIVABLE_MIN: Union[float, None]
    CC_AMT_TOTAL_RECEIVABLE_MAX: Union[float, None]
    CC_AMT_TOTAL_RECEIVABLE_MEAN: Union[float, None]
    CC_AMT_TOTAL_RECEIVABLE_SUM: Union[float, None]
    CC_AMT_TOTAL_RECEIVABLE_VAR: Union[float, None]
    CC_CNT_DRAWINGS_ATM_CURRENT_MIN: Union[float, None]
    CC_CNT_DRAWINGS_ATM_CURRENT_MAX: Union[float, None]
    CC_CNT_DRAWINGS_ATM_CURRENT_MEAN: Union[float, None]
    CC_CNT_DRAWINGS_ATM_CURRENT_SUM: Union[float, None]
    CC_CNT_DRAWINGS_ATM_CURRENT_VAR: Union[float, None]
    CC_CNT_DRAWINGS_CURRENT_MIN: Union[float, None]
    CC_CNT_DRAWINGS_CURRENT_MAX: Union[float, None]
    CC_CNT_DRAWINGS_CURRENT_MEAN: Union[float, None]
    CC_CNT_DRAWINGS_CURRENT_SUM: Union[float, None]
    CC_CNT_DRAWINGS_CURRENT_VAR: Union[float, None]
    CC_CNT_DRAWINGS_OTHER_CURRENT_MIN: Union[float, None]
    CC_CNT_DRAWINGS_OTHER_CURRENT_MAX: Union[float, None]
    CC_CNT_DRAWINGS_OTHER_CURRENT_MEAN: Union[float, None]
    CC_CNT_DRAWINGS_OTHER_CURRENT_SUM: Union[float, None]
    CC_CNT_DRAWINGS_OTHER_CURRENT_VAR: Union[float, None]
    CC_CNT_DRAWINGS_POS_CURRENT_MIN: Union[float, None]
    CC_CNT_DRAWINGS_POS_CURRENT_MAX: Union[float, None]
    CC_CNT_DRAWINGS_POS_CURRENT_MEAN: Union[float, None]
    CC_CNT_DRAWINGS_POS_CURRENT_SUM: Union[float, None]
    CC_CNT_DRAWINGS_POS_CURRENT_VAR: Union[float, None]
    CC_CNT_INSTALMENT_MATURE_CUM_MIN: Union[float, None]
    CC_CNT_INSTALMENT_MATURE_CUM_MAX: Union[float, None]
    CC_CNT_INSTALMENT_MATURE_CUM_MEAN: Union[float, None]
    CC_CNT_INSTALMENT_MATURE_CUM_SUM: Union[float, None]
    CC_CNT_INSTALMENT_MATURE_CUM_VAR: Union[float, None]
    CC_SK_DPD_MIN: Union[float, None]
    CC_SK_DPD_MAX: Union[float, None]
    CC_SK_DPD_MEAN: Union[float, None]
    CC_SK_DPD_SUM: Union[float, None]
    CC_SK_DPD_VAR: Union[float, None]
    CC_SK_DPD_DEF_MIN: Union[float, None]
    CC_SK_DPD_DEF_MAX: Union[float, None]
    CC_SK_DPD_DEF_MEAN: Union[float, None]
    CC_SK_DPD_DEF_SUM: Union[float, None]
    CC_SK_DPD_DEF_VAR: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Active_MIN: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Active_MAX: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Active_MEAN: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Active_SUM: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Active_VAR: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Approved_MIN: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Approved_MAX: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Approved_MEAN: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Approved_SUM: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Approved_VAR: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Completed_MIN: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Completed_MAX: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Completed_MEAN: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Completed_SUM: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Completed_VAR: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Demand_MIN: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Demand_MAX: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Demand_MEAN: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Demand_SUM: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Demand_VAR: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Refused_MIN: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Refused_MAX: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Refused_MEAN: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Refused_SUM: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Refused_VAR: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Sentproposal_MIN: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Sentproposal_MAX: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Sentproposal_MEAN: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Sentproposal_SUM: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Sentproposal_VAR: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Signed_MIN: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Signed_MAX: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Signed_MEAN: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Signed_SUM: Union[float, None]
    CC_NAME_CONTRACT_STATUS_Signed_VAR: Union[float, None]
    CC_NAME_CONTRACT_STATUS_nan_MIN: Union[float, None]
    CC_NAME_CONTRACT_STATUS_nan_MAX: Union[float, None]
    CC_NAME_CONTRACT_STATUS_nan_MEAN: Union[float, None]
    CC_NAME_CONTRACT_STATUS_nan_SUM: Union[float, None]
    CC_NAME_CONTRACT_STATUS_nan_VAR: Union[float, None]
    CC_COUNT: Union[float, None]


# 2. Class which describes a single flower measurements
class bank_model_old(BaseModel):
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

    data_df = pd.DataFrame.from_dict(data_dict, orient = 'index').T

    prediction = LGBM_model.predict([data_df.iloc[0].to_list()])
    return {
        'prediction': int(prediction[0]),
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn app:app --reload