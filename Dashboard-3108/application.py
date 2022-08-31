import pandas as pd
import numpy as np
import seaborn as sns
from lightgbm import LGBMClassifier
import requests
import pickle
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

# Solvabilité client sous forme de pie

def client(df, df_client):
    st.set_option('deprecation.showPyplotGlobalUse', False)

#  Appel de l'API
    url = "https://haby-p7-app.herokuapp.com/"   #"lien heroku"

    explainer_shap = -0.4203389942575416

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
    
# Information relative à un client 

option_sk = st.selectbox('Selectionner un numero de client',list_client_id)

row_df_sk = ( df['SK_ID_CURR'] == int(option_sk))
row_appli_sk = ( df_client['SK_ID_CURR'] == int(option_sk))

st.subheader("Client Information")
sex = df_client.loc[row_appli_sk, ['CODE_GENDER']].values[0][0]
st.write("Sex :",sex)
age = int(np.trunc(- int(df_client.loc[row_appli_sk, ['DAYS_BIRTH']].values)/365))
st.write("Age :", age)
family = df_client.loc[row_appli_sk, ['NAME_FAMILY_STATUS']].values[0][0]
st.write("Family status :", family)
education = df_client.loc[row_appli_sk, ['NAME_EDUCATION_TYPE']].values[0][0]
st.write("Education type :", education)
occupation = df_client.loc[row_appli_sk, ['OCCUPATION_TYPE']].values[0][0]
st.write("Occupation type :", occupation)
Own_realty = df_client.loc[row_appli_sk, ['FLAG_OWN_REALTY']].values[0][0]
st.write("Client owns a house or flat :", Own_realty)
income = str(df_client.loc[row_appli_sk, ['AMT_INCOME_TOTAL']].values[0][0])
st.write("Income of the client :", income)
income_perc = df_client.loc[row_df_sk, ['ANNUITY_INCOME_PERC']].values[0][0]
st.write(f"Loan annuity / Income of the client : {income_perc*100:.2f} %")

st.subheader("Credit Information")
type_contract = str(df_client.loc[row_appli_sk, ['NAME_CONTRACT_TYPE']].values[0][0])
st.write("Contract type :", type_contract)
credit = str(df_client.loc[row_appli_sk, ['AMT_CREDIT']].values[0][0])
st.write("Credit amount of the loan :", credit)
annuity = df_client.loc[row_appli_sk, ['AMT_ANNUITY']].values[0][0] / 12
st.write(f"Loan monthly : {annuity:.1f}")
income_credit_perc = df_client.loc[row_df_sk, ['INCOME_CREDIT_PERC']].values[0][0]
st.write(f"Income of the client / Credit amount of the loan : {income_credit_perc*100:.2f} %")


st.subheader("Retour Prediction")
st.write("""
    **le retour est un score de 0 à 100. Le seuil de refus est à 50.**
    
    1. Un retour en dessous de 40 est une acceptation du crédit.
    
    2. Un retour au dessus de 60 est un refus du crédit.
    
    3. Pour un score entre 40 et 60, on va regarder l'interpretabilité de la prediction pour aider au choix. 
    
    """)
    
    
fig = update_sk(option_sk)
st.plotly_chart(fig)

st.subheader("Feature importance")

 #Feature importance / description
original_title = '<p style="font-size: 20px;text-align: center;"> <u>Quelles sont les informations les plus importantes dans la prédiction ?</u> </p>'
st.markdown(original_title, unsafe_allow_html=True)
feature_imp = pd.DataFrame(sorted(zip(grid_lgbm.booster_.feature_importance(importance_type='gain'), df.columns)), columns=['Value','Feature'])
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(5))
ax.set(title='Importance des informations', xlabel='', ylabel='')
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