import imblearn
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

#basically stuff in the preprocessing notebook
def preprocessing(input):
    df = pd.read_csv(input)
    
    #getting only the ones with Good/Bad columns filled in
    if "Good/Bad" in df.columns:
        df = df[(df["Good/Bad"] == -1) | (df["Good/Bad"] == 1)]
    
    #dropping the ones with two many NAs
    len = df.shape[0];
    df = df.dropna(axis=1, thresh = len/2)
    
    #getting numeric cols (imputer only uses numeric not categorical)
    numeric_cols = df.select_dtypes(include=np.number).columns
    numeric_cols = numeric_cols[:-1].tolist()
    
    #imputer time, replacing with mean
    imputer = SimpleImputer(strategy="mean")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    df[numeric_cols] = pd.DataFrame(df, columns=numeric_cols)

    #removing the cols with all zeros
    cols = []
    for col in df.columns:
        if (df[col] == 0).all():
            cols.append(col)

    if cols:
        df.drop(columns=cols, inplace=True)
        
    #SMOTE time (if they provided a Good/Bad col)
    if "Good/Bad" in df.columns:
        has_col = True
        X = df.iloc[:, :-1]
        Y = df["Good/Bad"]
        
        smt = SMOTE()
        X_resampled, y_resampled = smt.fit_resample(X, Y)
        #balancing the df
        df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='Good/Bad')], axis=1)
    else:
        has_col = False
    
    #Reducing the data to just the necessary features and returning
    print(df.columns)
    high_corr_features = ['Sensor-57', 'Sensor-134', 'Sensor-76', 'Sensor-28', 'Sensor-164', 'Sensor-369', 'Sensor-108', 'Sensor-81', 'Sensor-449', 'Sensor-319']
    if np.in1d(high_corr_features, df.columns).all():
        error = False
        high_corr_features.append('Good/Bad')
        df = df[high_corr_features]
        # print(f"yoo {df.columns}")
        return df, has_col, error
    else:
        error = True
        return df, has_col, error 
