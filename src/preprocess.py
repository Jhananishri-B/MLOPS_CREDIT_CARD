import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, RobustScaler

def load_data(path=r"D:\AI WORKSHOP\TASK\MLOPS\CC_GENERAL_preprocessed.csv"):
    return pd.read_csv(path)

def clean_data(df):
    df['CREDIT_LIMIT'] = df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median())
    df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median())
    df = df.drop(['CUST_ID'], axis=1, errors='ignore')
    return df

def transform_data(df):
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    df_transformed = pd.DataFrame(pt.fit_transform(numeric_df), columns=numeric_df.columns)
    return df_transformed

def scale_data(df, features):
    scaler = RobustScaler()
    df_scaled = scaler.fit_transform(df[features])
    return df_scaled, scaler
