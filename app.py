import io
import json                    
import base64              
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, abort
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

class output:
    def __init__(self,mse,r2,pred,graphs) -> None:
        self.mse = mse
        self.r2 = r2
        self.pred = pred
        self.graphs = graphs
    def getJSON(self):
        return json.dumps({"mse":self.mse,"r2":self.r2,"pred":(self.pred).to_dict(),"graphs":self.graphs})

model = pickle.load(open('finalized_model.sav','rb'))


app = Flask(__name__)          

@app.route("/test", methods=['POST'])        
def Random_Forest():
    if not request.json:
        abort(400)
    # print(type(payload))
    data_dic=request.json
    train_ratio=data_dic.pop("train_ratio")
    df1=pd.DataFrame(data_dic)
    new_cust_deets=(df1.iloc[-1]).drop(['cost'])
    
    # last_row = len(df1)
    # df1=df1.drop(df.index[60428],axis=0)
    
    df1=df1.drop([df1.index[-1]])
    
    
    X = df1.drop(columns='cost')
    y = df1['cost']
    
    n_rows = df1.shape[0]
    train_rows = int(n_rows * train_ratio)
    X_train = X[:train_rows]
    y_train = y[:train_rows]
    X_test = X[train_rows:]
    y_test = y[train_rows:]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    
        
    
    new_cust_cac=pd.DataFrame(rf.predict([new_cust_deets]))
    x="graphs"
    
    return (output(mean_squared_error(y_test, y_pred),r2_score(y_test, y_pred),new_cust_cac,x).getJSON())



def run_server_api():
    app.run(host='0.0.0.0', port=8080)

if __name__ == "__main__":     
    run_server_api()