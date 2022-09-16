import pandas as pd
import numpy as np
import seaborn as sns
from lightgbm import LGBMClassifier
import requests
import pickle
import shap
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Chargement des données
df = pd.read_csv('data_scoring_ech.csv', index_col=0)
df_client= pd.read_csv('df_client.csv',index_col=0)
df_appli=df.copy()

df_appli['NAME_FAMILY_STATUS']=df_client['NAME_FAMILY_STATUS']
df_appli['NAME_EDUCATION_TYPE']=df_client['NAME_EDUCATION_TYPE']
df_appli['OCCUPATION_TYPE']=df_client['OCCUPATION_TYPE']
df_appli['NAME_CONTRACT_TYPE']=df_client['NAME_CONTRACT_TYPE']
df_appli['CODE_GENDER']=df_client['CODE_GENDER']
df_appli['DAYS_BIRTH']=df_client['DAYS_BIRTH']
df_appli['FLAG_OWN_REALTY']=df_client['FLAG_OWN_REALTY']
df_appli['AMT_INCOME_TOTAL']=df_client['AMT_INCOME_TOTAL']
df_appli['ANNUITY_INCOME_PERC']=df_client['ANNUITY_INCOME_PERC']
df_appli['AMT_CREDIT']=df_client['AMT_CREDIT']
df_appli['AMT_ANNUITY']=df_client['AMT_ANNUITY']
df_appli['INCOME_CREDIT_PERC']=df_client['INCOME_CREDIT_PERC']

#chargement du modéle
grid_lgbm = pickle.load( open( 'lgbm_GridCV.p', 'rb' ) )

#liste client
#list_client_id=df.SK_ID_CURR.unique()
#list_client_id=list(num_client.astype(str))

# Solvabilité client sous forme de pie

#def client(df, df_client):
st.set_option('deprecation.showPyplotGlobalUse', False)

#  Appel de l'API
url = "https://p7api.herokuapp.com/"   #"lien heroku"

url_requests = url+"predict/"
response = requests.get(url_requests)
if response:
     list_client_id = response.json()['list_client_id']
     list_client_id = sorted(list_client_id)
else:
     list_client_id = ['000000']
     print("erreur web : ", response)
def update_sk(sk_id):
        predict_proba_1=0.5
        if sk_id in list_client_id:
            url_pred = url + "predict/" + sk_id
            response = requests.get(url_pred)
            if response:
                predict_proba_1 = float(response.json()['predict_proba_1'])
            else:
                print("erreur web : ", response)

        gauge_predict = go.Figure(go.Indicator( mode = "gauge+number",
                                                value = predict_proba_1*100,
                                                domain = {'x': [0, 1], 'y': [0, 1]},
                                                gauge = {
                                                    'axis': {'range': [0, 100], 'tickwidth': 0.2, 'tickcolor': "darkblue"},
                                                    'bgcolor': "lightcoral",
                                                    'steps': [
                                                        {'range': [0, 40], 'color': 'lightgreen'},
                                                        {'range': [40, 60], 'color': 'palegoldenrod'}
                                                    ],
                                                    'threshold': {
                                                        'line': {'color': "red", 'width': 4},
                                                        'thickness': 0.75,
                                                        'value': 100}},
                                                title = {'text': f"client {sk_id}"}))

        return gauge_predict
    
#affichage formulaire
original_title = '<p style="font-family:Courier; color:Red; font-size: 50px;text-align: center;">Accord prêt bancaire</p>'
st.markdown(original_title, unsafe_allow_html=True)
st.markdown("***")
st.write("""Cette application prédit la probabilité qu'un client de la banque "Prêt à dépenser" ne rembourse pas son prêt.
""")

seuil= 0.6
original_title = '<p style="font-family:Courier; color:Blue; font-size: 18px;">La probabilité maximale de défaut de remboursement autorisée par la banque est de : {}</p>'.format(seuil)
st.markdown(original_title, unsafe_allow_html=True)

option_sk = st.selectbox('Selectionner un numero de client',list_client_id)
id_client = option_sk

# Information relative à un client 
row_df_sk =  df[df.SK_ID_CURR == int(id_client)] 
row_appli_sk = df_appli[df_appli['SK_ID_CURR'] == int(id_client)]

st.subheader("Client Information")
st.write("**Sex :**", row_appli_sk['CODE_GENDER'].values[0])
st.write("**Age client :**", row_appli_sk["DAYS_BIRTH"].values[0] , "ans")
st.write("**Family status :**", row_appli_sk['NAME_FAMILY_STATUS'].values[0])
st.write("**Education type :**", row_appli_sk['NAME_EDUCATION_TYPE'].values[0])
st.write("**Occupation type :**", row_appli_sk['OCCUPATION_TYPE'].values[0])
st.write("**Client owns a house or flat :**", row_appli_sk['FLAG_OWN_REALTY'].values[0])
st.write("**Income of the client :**", row_appli_sk['AMT_INCOME_TOTAL'].values[0])
st.write("**ANNUITY_CREDIT_PERCENT_INCOME :** {:.2f}".format(row_appli_sk['ANNUITY_INCOME_PERC'].values[0]*100), "%")


st.subheader("Credit Information")
st.write("**Contract type :**", row_appli_sk['NAME_CONTRACT_TYPE'].values[0])
st.write("**Credit amount of the loan :**", row_appli_sk['AMT_CREDIT'].values[0])
annuity =row_appli_sk['AMT_ANNUITY'].values[0] / 12
st.write("**Loan monthly :** {:.1f}".format(annuity))
st.write("**INCOME_CREDIT_PERC :** {:.2f}".format(row_df_sk['INCOME_CREDIT_PERC'].values[0]*100), "%")
#income_credit_perc =row_df_sk['INCOME_CREDIT_PERC'].values[0]
#st.write(f"Income of the client / Credit amount of the loan : {income_credit_perc*100:.2f} %")

#affichage de la prédiction
st.subheader("Retour Prediction")
st.write("""
    **le retour est un score de 0 à 100. Le seuil de refus est à 50.**
    
    1. Un retour en dessous de 40 est une acceptation du crédit.
    
    2. Un retour au dessus de 60 est un refus du crédit.
    
    3. Pour un score entre 40 et 60, on va regarder l'interpretabilité de la prediction pour aider au choix. 
    
    """)
    
    
fig = update_sk(id_client)
st.plotly_chart(fig)

#Feature importance / description
st.subheader("Feature importance")                           
shap.initjs()  
X=df[df['SK_ID_CURR']==int(id_client)]
#X.TARGET=df_TAREGT
            
fig, ax = plt.subplots(figsize=(10, 10))
explainer = shap.TreeExplainer(grid_lgbm)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, features=X, plot_type ="bar", max_display=10, color_bar=False, plot_size=(10, 10))            
#shap.bar_plot(shap_values[0],feature_names=np.array(feats),max_display=10)            
st.pyplot(fig)    

st.subheader("Clients similaires")

nearest_neighbors = df_appli[['SK_ID_CURR','AMT_INCOME_TOTAL', 'AMT_ANNUITY','CODE_GENDER', 'AMT_CREDIT',
                              'INCOME_CREDIT_PERC','DAYS_BIRTH']]

knn = KMeans(random_state=42,n_clusters=5) #5 plus proche voisins
knn.fit(nearest_neighbors)
# Creation de nouvelle feature
nearest_neighbors['class']=knn.labels_
row_client=nearest_neighbors[nearest_neighbors.SK_ID_CURR == int(id_client)]
row_client['CODE_GENDER'] = row_client['CODE_GENDER'].map({0:'Men',1:'Women'})
st.table(row_client)
#5 client de la même classe par hazard
#cls = nearest_neighbors[nearest_neighbors.SK_ID_CURR == int(id_client)]['class']
#k = nearest_neighbors['class'][nearest_neighbors['class'] == cls.values[0]].sample(5)
affiche_voisin = nearest_neighbors[['DAYS_BIRTH', 'AMT_CREDIT','AMT_INCOME_TOTAL', 'AMT_ANNUITY','CODE_GENDER','INCOME_CREDIT_PERC','class']]
affiche_voisin['DAYS_BIRTH']=np.round(affiche_voisin['DAYS_BIRTH'],0)
affiche_voisin['CODE_GENDER'] = affiche_voisin['CODE_GENDER'].map({0:'Men',1:'Women'})
affiche_voisin.head()
voisin_similaire=affiche_voisin[affiche_voisin['class']==row_client['class'].values[0]]
st.write(voisin_similaire.head(5))
 
