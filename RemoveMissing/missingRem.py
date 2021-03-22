import numpy as np
import pandas as pd
from missingpy import MissForest

df = pd.read_csv('D:\Final year project\Dataset\Original\Veh.csv')

#df = pd.read_csv(r'C:\Users\Aravindsagar P A\Downloads\abc.csv')  #testing

'''
#only needed for acc.csv
df = df.drop(columns = ['Accident_Index'])
df = df.drop(columns = ['LSOA_of_Accident_Location'])
df['Date'] = df['Date'].str.replace("-","").astype(int)
df['Time'] = df['Time'][df['Time'].notnull()].str.replace(":","").astype(int)
df.drop(df[df['Local_Authority_(Highway)'] == 'EHEATHROW'].index, inplace = True) 
df['Local_Authority_(Highway)'] = df['Local_Authority_(Highway)'][df['Local_Authority_(Highway)'].notnull()].str.replace('[A-Z]','').astype(int)
'''
#missForrest method---------------------------
'''
print("---------training--------")

imputer = MissForest()
#X = df.drop('Accident_Severity', axis=1)  #for acc.csv
X = df.drop('Casualty_Severity', axis=1)   #for cas.csv
X_imputed = imputer.fit_transform(X)

df_org = df.copy()
#df = df.drop('Accident_Severity', axis=1)  #for acc.csv
df = df.drop('Casualty_Severity', axis=1)   #for cas.csv
for i in range(df.shape[0]):
    df.iloc[i,:] = X_imputed[i]
#df['Accident_Severity'] = df_org['Accident_Severity'] #for acc.csv
df['Casualty_Severity'] = df_org['Casualty_Severity']  #for cas.csv



#output csv
out = pd.DataFrame(data = df)
out.to_csv('xyz.csv')
'''

print(df.isnull().sum())

