# Data Prepare of Credit card 
import pandas as pd


df = pd.read_csv('../data/credit-data.csv')
df.info()

# Metodos uteis
# preenche com a mediana uma coluna AGRUPADA
def fill_na_median(dataframe, grupo, valor, tipo='median'):
    return dataframe[valor].fillna(dataframe.groupby(grupo)[valor].transform(tipo))
#normaliza a coluna do tipo standardization
def nomaliza_std(dataframe, coluna):
    return (dataframe[coluna]-dataframe[coluna].mean())/dataframe[coluna].std()

# fill na values with median of serie
df['age'] = df['age'].fillna(df['age'][df['age']>0].median())
# replace negative velues with median
df.loc[df['age']<0, 'age'] = df['age'][df['age']>0].median()

# normalize columns 
for coluna in ['income', 'age', 'loan']:
    df[coluna] = nomaliza_std(df, coluna)

