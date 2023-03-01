import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('dataset_phishing.csv')
value_counts = df['status'].value_counts()

print(value_counts)

print(df.iloc[1]['https_token'], df.iloc[1]['url'])

df_category =df[['https_token','punycode','port','ip','abnormal_subdomain','random_domain','dns_record','google_index']]

#print(df.columns)

for i in df_category.columns:
    
    if i != 'status' and i != 'length_url':

        legitimate = len(df[(df[i] == 0) & (df['status'] == 'legitimate')])
        phishing = len(df[(df[i] == 0) & (df['status'] == 'phishing')])

        if not (legitimate == phishing and legitimate == 0):
            print("")
            print("%s analysis" % i)
            print("Legitimate sites with %s: %f %%" % (i, round(legitimate / value_counts[0]* 100, 2)))
            print("Phishing sites with %s: %f %%" % (i, round(phishing / value_counts[1]*100, 2)))
    

df.loc[df['status'] == 'phishing', 'status'] = 1
df.loc[df['status'] == 'legitimate', 'status'] = 0

#corr = df.corr()
#print(corr.columns)



df_without_url = df.drop('url', axis=1)
#print(df_without_url['status'])
#print(df.columns)

"""
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
    plt.show()
"""  



"""
HTTPS
Punycode
Port
#subdomain
abnormal subdomains
prefix suffix
random domains

Phish hints
External CSS
"""
