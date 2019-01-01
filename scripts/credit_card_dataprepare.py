# Data Prepare of Credit card 
from pandas import read_csv
from pandas import DataFrame
import matplotlib.pyplot as plt
from seaborn import heatmap 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


# Metodos uteis
# preenche com a mediana uma coluna AGRUPADA
def fill_na_median(dataframe, grupo, valor, tipo='median'):
    return dataframe[valor].fillna        (dataframe.groupby(grupo)[valor]         .transform(tipo))
#normaliza a coluna do tipo standardization
def nomaliza_std(dataframe, coluna):
    return (dataframe[coluna]-dataframe[coluna].mean())/dataframe[coluna].std()
# adapted of : 
# https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
def plt_confusion_matrix(confusion_matrix,class_names, figsize = (10,7), fontsize=14):
    df_cm = DataFrame(
            confusion_matrix, index=class_names, columns=class_names, 
        )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap_ = heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap_.yaxis.set_ticklabels(heatmap_.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap_.xaxis.set_ticklabels(heatmap_.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

df = read_csv('../data/credit-data.csv')

# fill na values with median of serie
df['age'] = df['age'].fillna(df['age'][df['age']>0].median())
# replace negative velues with median
df.loc[df['age']<0, 'age'] = df['age'][df['age']>0].median()
# normalize columns 
for coluna in ['income', 'age', 'loan']:
    df[coluna] = nomaliza_std(df, coluna)

#X = df[['income', 'age', 'loan']]
#y = df['default']    
X_train, X_test, y_train, y_test = train_test_split(df[['income', 'age', 'loan']],
                                                   df['default'],
                                                   test_size=0.2,
                                                   random_state=0)



naive_classifier = GaussianNB()
naive_classifier.fit(X_train, y_train)


predictions = naive_classifier.predict(X_test)


cnf_matrix = confusion_matrix(y_test, predictions, labels=[1,0])
plt_confusion_matrix(cnf_matrix, class_names=['Paga', 'Nao paga'])





