import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
import matplotlib.pyplot as plt
import seaborn as sns
from KNN import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def Data_clean():

    # Lectura de datos
    df = pd.read_csv('dataset_phishing.csv')
    scaler = StandardScaler()

    #Transformación de datos
    df.loc[df['status'] == 'phishing', 'status'] = -1
    df.loc[df['status'] == 'legitimate', 'status'] = 1
    df = df.drop('url', axis=1)
    df_temp = df
    df['status'] = df['status'].astype(int)
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    #Cantidad de elementos por clase
    value_counts = df['status'].value_counts()
    
    return df,df_temp
    
def Analisis_exploratorio(df):

    #Distribución de los datos
    with plt.style.context(style="fivethirtyeight"):
        plt.pie(x=dict(df['status'].value_counts()).values(),
            labels=dict(df['status'].value_counts()).keys(),
            autopct="%.2f%%",
            colors=['red','orangered'],
            startangle=90,
            explode=[0,0.05])
        centre_circle=plt.Circle((0,0),0.70,fc='white')
        fig=plt.gcf()
        fig.gca().add_artist(centre_circle)
        plt.title(label="Analysing status feature using donut-chart")
        plt.show( block = True)

    corr = df.corr()['status']
    columns = corr[corr > 0.1].index.tolist()
    print("\n Las variables a utilizar son: ")
    a=""
    for c in columns:
        a+=c+", "

    print(a)
    print("---------------------")

    linear_coefficients = []
    model = LinearRegression()
    df_without_status = df.drop('status', axis=1)
    for column in df_without_status.columns:
        model.fit(df[['status']], df[column])
        if model.score(df[['status']], df[column]) > 0.2:
            linear_coefficients.append(column)

    columns = list(set(linear_coefficients + columns + ['status']))
    df = df[columns]
    
    return df





"""
df_category =df[['https_token','punycode','port','ip','abnormal_subdomain','random_domain','dns_record','google_index']]
for i in df_category.columns:
    
    if i != 'status' and i != 'length_url':

        legitimate = len(df[(df[i] == 0) & (df['status'] == 'legitimate')])
        phishing = len(df[(df[i] == 0) & (df['status'] == 'phishing')])

        if not (legitimate == phishing and legitimate == 0):
            print("")
            print("%s analysis" % i)
            print("Legitimate sites with %s: %f %%" % (i, round(legitimate / value_counts[0]* 100, 2)))
            print("Phishing sites with %s: %f %%" % (i, round(phishing / value_counts[1]*100, 2)))
 
"""



