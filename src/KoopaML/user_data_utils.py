#Import libraries
import pandas as pd

def load_database():
    df = pd.read_excel("~/projects/DoubleAntiaggregation/data/PACS_DAPT_preprocess.xlsx")
    return df

def clean_database(df):
    return df

def process_database(df):
    df = df.loc[(df.Registro == 'BLEEMACS')|(df.Registro == 'RENAMI'),]
    return df

def fillna_database(df):
    df=df.fillna(0)
    return df

def preprocess_filtered_database(df, wf_name):
    return df
