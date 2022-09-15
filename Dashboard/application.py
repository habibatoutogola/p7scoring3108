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
df_TARGET = df.TARGET.copy()
df.drop(columns='TARGET', inplace=True)

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

    #explainer_shap = -0.4203389942575416

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
st.title('Dashboard Scoring Credit')
st.markdown("Prédictions de scoring client, notre seuil de choix est de 40 %")
option_sk = st.selectbox('Selectionner un numero de client',list_client_id)
id_client = option_sk

# Information relative à un client 
row_df_sk =  df[df.SK_ID_CURR == int(id_client)] 
row_appli_sk = df_client[df_client['SK_ID_CURR'] == int(id_client)]
#st.table(row_appli_sk)

st.subheader("Client Information")
st.write("**Sex :**", row_appli_sk['CODE_GENDER'].values[0])
st.write("**Age client :**", row_appli_sk["DAYS_BIRTH"].values[0] , "ans.")
st.write("**Family status :**", row_appli_sk['NAME_FAMILY_STATUS'].values[0])
st.write("**Education type :**", row_appli_sk['NAME_EDUCATION_TYPE'].values[0])
st.write("**Occupation type :**", row_appli_sk['OCCUPATION_TYPE'].values[0])
st.write("**Client owns a house or flat :**", row_appli_sk['FLAG_OWN_REALTY'].values[0])
st.write("**Income of the client :**", row_appli_sk['AMT_INCOME_TOTAL'].values[0])
st.write("**ANNUITY_CREDIT_PERCENT_INCOME :**{:.2f}".format(row_df_sk['ANNUITY_INCOME_PERC'].values[0]*100), "%")
#income_perc =row_df_sk['ANNUITY_INCOME_PERC'].values
#st.write(f"Loan annuity / Income of the client : {income_perc*100:.2f} %")

st.subheader("Credit Information")
st.write("**Contract type :**", row_appli_sk['NAME_CONTRACT_TYPE'].values[0])
st.write("**Credit amount of the loan :**", row_appli_sk['AMT_CREDIT'].values[0])
annuity =row_appli_sk['AMT_ANNUITY'].values[0] / 12
st.write("**Loan monthly :**{:.1f}".format(annuity))
st.write("**INCOME_CREDIT_PERC :**{:.2f}".format(row_df_sk['INCOME_CREDIT_PERC'].values[0]*100), "%")
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
X.TARGET=df_TAREGT
            
fig, ax = plt.subplots(figsize=(10, 10))
explainer = shap.TreeExplainer(grid_lgbm)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, features=X, plot_type ="bar", max_display=10, color_bar=False, plot_size=(10, 10))            
#shap.bar_plot(shap_values[0],feature_names=np.array(feats),max_display=10)            
st.pyplot(fig)              

st.subheader("Client similaires")

nearest_neighbors=df_client.copy()
knn = KMeans(random_state=42,n_clusters=5) #5 plus proche voisins
knn.fit(nearest_neighbors)
# Creation de nouvelle feature
nearest_neighbors['class']=knn.labels_
affiche_voisin = nearest_neighbors[['DAYS_BIRTH', 'AMT_CREDIT','AMT_INCOME_TOTAL', 'AMT_ANNUITY','CODE_GENDER','INCOME_CREDIT_PERC']]
affiche_voisin['DAYS_BIRTH']=np.round(affiche_voisin['DAYS_BIRTH'],0)
affiche_voisin['CODE_GENDER'] = affiche_voisin['CODE_GENDER'].map({0:'Men',1:'Women'})
affiche_voisin.head()
st.dataframe(affiche_voisin)


# Information relative à un client 

option_sk = st.selectbox('Selectionner un numero de client',list_client_id)
id_client = st.text_input("Veuillez entrer l'Identifiant d'un client")

row_df_sk =  df[df['SK_ID_CURR'] == id_client] 
row_appli_sk = df_client[df_client['SK_ID_CURR'] == id_client]
st.subheader("Client Information")
st.write("Sex :", row_appli_sk['CODE_GENDER'].values[0][0])
st.write("Age client :", row_appli_sk["DAYS_BIRTH"].values[0] , "ans.")
st.write("Family status :", row_appli_sk['NAME_FAMILY_STATUS'].values[0], "$")
st.write("Education type :", row_appli_sk['NAME_EDUCATION_TYPE'].values[0], "$")
st.write("Occupation type :", row_appli_sk['OCCUPATION_TYPE'].values[0], "$")
st.write("Client owns a house or flat :", row_appli_sk['FLAG_OWN_REALTY'].values[0], "$")
st.write("Income of the client :", row_appli_sk['AMT_INCOME_TOTAL'].values[0], "$")
income_perc =row_df_sk['ANNUITY_INCOME_PERC'].values[0]
st.write(f"Loan annuity / Income of the client : {income_perc*100:.2f} %")
st.write("Sex :", row_appli_sk['CODE_GENDER'].values[0], "$")
st.write("Sex :", row_appli_sk['CODE_GENDER'].values[0], "$")
st.write("Sex :", row_appli_sk['CODE_GENDER'].values[0], "$")

st.subheader("Credit Information")
st.write("Contract type :", row_appli_sk['Contract type'].values[0], "$")
st.write("Credit amount of the loan :", row_appli_sk['AMT_CREDIT'].values[0], "$")
annuity =row_appli_sk['AMT_ANNUITY'].values[0] / 12
st.write(f"Loan monthly : {annuity:.1f}")
income_credit_perc =row_df_sk['INCOME_CREDIT_PERC'].values[0]
st.write(f"Income of the client / Credit amount of the loan : {income_credit_perc*100:.2f} %")
