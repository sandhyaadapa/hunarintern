import pandas as pd
#load the dataset
df=pd.read_csv("food_coded.csv")
print("BEFORE CLEANING")
print(df.info())
print(df.isnull().sum())
#Remove exact duplicate rows
df.drop_duplicates(inplace=True)
#Remove duplicate columns
df=df.loc[:,~df.columns.duplicated()]
#convert GPA column to numeric
df['GPA']=pd.to_numeric(df['GPA'],errors='coerce')
#fill missing values
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        #fill numeric columns with madian(no inplace)
        df[col]=df[col].fillna(df[col].median())
    else:
        #fill categorical columns with mode(no place)
        mode=df[col].mode()
        if not mode.empty:
            df[col]=df[col].fillna(mode[0])
#final check after cleaning
print("AFTER CLEANING")
print(df.info())
print(df.isnull().sum())
