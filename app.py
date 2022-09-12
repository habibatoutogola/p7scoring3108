from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import pickle

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


app = Flask(__name__)

#with open('lgbmc10_feats_custom.p', 'rb') as f2:
with open('lgbm_GridCV.p', 'rb') as f2:
    print("utilisation modele lgbm_GridCV")
    grid_lgbm = pickle.load(f2)

df = pd.read_csv('data_scoring_ech.csv', index_col=0)
num_client = df.SK_ID_CURR.unique()


@app.route('/')
def home():
    return ('Prevision credit')


@app.route('/predict/')
def predict():
    """
    Returns
    liste des clients dans le fichier
    """
    return jsonify({"model": "'lgbmc10_GridCV",
                    "list_client_id" : list(num_client.astype(str))})


@app.route('/predict/<int:sk_id>')
def predict_get(sk_id):
    """
    Parameters
    ----------
    sk_id : numero de client
    Returns
    -------
    prediction  0 pour paiement OK
                1 pour defaut de paiement
    """

    if sk_id in num_client:
        predict = grid_lgbm.predict(df[df['SK_ID_CURR']==sk_id])[0]
        predict_proba = grid_lgbm.predict_proba(df[df['SK_ID_CURR']==sk_id])[0]
        predict_proba_0 = str(predict_proba[0])
        predict_proba_1 = str(predict_proba[1])
    else:
        predict = predict_proba_0 = predict_proba_1 = "client inconnu"
    return jsonify({ 'retour_prediction' : str(predict), 'predict_proba_0': predict_proba_0,
                     'predict_proba_1': predict_proba_1 })


if __name__ == '__main__':
        app.run()
