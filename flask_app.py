from flask import Flask, render_template
import sys
import os
import json
import pandas
import numpy
from pandas.io.json import json_normalize
import pickle
from sklearn.ensemble import AdaBoostClassifier

FILEPATH=os.path.realpath(__file__)
ROOTPATH=os.path.split(FILEPATH)[0]
SRCPATH=os.path.join(ROOTPATH,'src')
DATAPATH=os.path.join(ROOTPATH,'data')
PREDICTJSON=os.path.join(DATAPATH,'predict.json')
TESTJSON=os.path.join(DATAPATH,'test_tom.json')
MODELPATH=os.path.join(ROOTPATH,'models')
sys.path.append(SRCPATH)
from riggi_model import transform_one
from clean_api_data import APIPipeline

def get_one(path):
    with open(path, 'r') as f:
        data=json.loads(f.read())
        return data[-1]


with open(os.path.join(MODELPATH,'riggi_model.pkl'), 'rb') as f:
    ada_boosted = pickle.load(f)
with open(os.path.join(MODELPATH,'lucas_RF_model_all_data.pkl'), 'rb') as f:
    random_forest = pickle.load(f)


app = Flask(__name__)

#home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/test1')
def test1():
    df=json_normalize(get_one(PREDICTJSON)).T
    df_p=transform_one(json_normalize(get_one(PREDICTJSON)))
    predict=ada_boosted.predict(df_p)[0]
    if predict==1:
        predict='FRAUD'
    else:
        predict='LEGIT'
    prob_predict=str(round(ada_boosted.predict_proba(df_p)[0].max(),3)*100)+'%'
    return render_template('test1.html',prob_predict=prob_predict,predict=predict,tables=[df.to_html(classes='data')], titles=df.columns.values)

@app.route('/test2')
def test2():
    df=json_normalize(get_one(PREDICTJSON)).T
    df_p=APIPipeline('api',get_one(PREDICTJSON)).df.values
    predict= random_forest.predict(df_p)
    if predict==1:
        predict='FRAUD'
    else:
        predict='LEGIT'
    prob_predict=str(round(random_forest.predict_proba(df_p)[0].max(),3)*100)+'%'
    return render_template('test2.html',prob_predict=prob_predict,predict=predict,tables=[df.to_html(classes='data')], titles=df.columns.values)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080,threaded=True)




