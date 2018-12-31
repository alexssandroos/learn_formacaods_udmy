from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from pandas import read_csv 
from pandas import get_dummies
from pandas import concat

df = read_csv('../data/census.csv')

#funcoes uteis
def fill_na_median(dataframe, grupo, valor, tipo='median'):
    return dataframe[valor].fillna(dataframe.groupby(grupo)[valor].transform(tipo))
#normaliza a coluna do tipo standardization
def nomaliza_std(dataframe, coluna):
    return (dataframe[coluna]-dataframe[coluna].mean())/dataframe[coluna].std()
#return columns with one hot encoding
def set_onehotencoding(dataframe, coluna):
    cols = get_dummies(dataframe[coluna], prefix=coluna, drop_first=False)
    dataframe.drop(coluna, axis=1, inplace=True)
    return concat([dataframe,cols],axis=1)


# uso de Label Encoder
'''
le = LabelEncoder()
features_to_encoder = ['workclass', 'education','marital-status',\
                       'occupation', 'relationship', 'race', 'sex', 'native-country']
for feature in features_to_encoder:
    df[feature] = le.fit_transform(df[feature])
'''


#uso do one hot encoder
features_to_encoder = ['workclass', 'education','marital-status',\
'occupation', 'relationship', 'race', 'sex', 'native-country']
for feature in features_to_encoder:
    df = set_onehotencoding(df, feature)

