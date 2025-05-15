import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    
    categorical_columns = ['condition', 'color_type']  
    
    for col in categorical_columns:
        df[col] = df[col].astype(str).fillna("Unknown")
        df[col] = df[col].astype(str)
        
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    return df