# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 10:26:20 2021

@author: Chintamani
"""
'''
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask("__name__",template_folder='template')

dataframe=pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
dataframe = dataframe.drop('Churn',axis=1)
dataframe = dataframe.drop('customerID',axis=1)

q = ""

@app.route("/")
def loadPage():
	return render_template('index.html', query="")


@app.route("/", methods=['POST'])
def predict():
    
    query_seniorcitizen = request.form['seniorcitizen']
    query_monthlycharges = request.form['monthlycharges']
    query_totalcharges = request.form['totalcharges']
    query_gender = request.form['gender']
    query_partner = request.form['partner']
    query_dependents = request.form['dependents']
    query_phoneservices = request.form['phoneservice']
    query_multiplelines = request.form['multiplelines']
    query_internerservice = request.form['internetservice']
    query_onlinesecurity = request.form['onlinesecurity']
    query_onlinebackup = request.form['onlinebackup']
    query_deviceprotection = request.form['deviceprotection']
    query_techsupport = request.form['techsupport']
    query_streamingtv = request.form['streamingtv']
    query_streamingmovies = request.form['streamingmovies']
    query_contract = request.form['contract']
    query_paperlessbilling = request.form['paperlessbilling']
    query_paymentmethod = request.form['paymentmethod']
    query_tenure = request.form['tenure']

    model = pickle.load(open("RDForest_model.sav", "rb"))
    
    input_data = [[query_seniorcitizen,
                   query_monthlycharges,
                   query_totalcharges,
                   query_gender,
                   query_partner,
                   query_dependents,
                   query_phoneservices,
                   query_multiplelines,
                   query_internerservice,
                   query_onlinesecurity,
                   query_onlinebackup,
                   query_deviceprotection,
                   query_techsupport,
                   query_streamingtv,
                   query_streamingmovies,
                   query_contract,
                   query_paperlessbilling,
                   query_paymentmethod,
                   query_tenure]]
    
    new_dataframe = pd.DataFrame(input_data, columns = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
                                           'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                           'PaymentMethod', 'tenure'])
    
    dataframe_2 = pd.concat([dataframe, new_dataframe], ignore_index = True) 
    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 24)]
    
    dataframe_2['tenure_group'] = pd.cut(dataframe_2.tenure.astype(int), range(1, 80, 24), right=False, labels=labels)
    #drop column customerID and tenure
    dataframe_2.drop(columns= ['tenure'], axis=1, inplace=True)   
    
    
    
    
    new_df__dummies = pd.get_dummies(dataframe_2[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
           'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
           'Contract', 'PaperlessBilling', 'PaymentMethod','tenure_group']])
    
    
    #final_df=pd.concat([new_df__dummies, new_dummy], axis=1)
        
    
    single = model.predict(new_df__dummies.tail(1))
    probablity = model.predict_proba(new_df__dummies.tail(1))[:,1]
    
    if single==1:
        o1 = "This customer is likely to be churned!! with "
        o2 = "Confidence: {}".format(probablity*100)
    else:
        o1 = "This customer is likely to continue!!"
        o2 = "Confidence: {}".format(probablity*100)
        
    return render_template('home.html', output1=o1, output2=o2, 
                           query_seniorcitizen = request.form['seniorcitizen'],
                           query_monthlycharges = request.form['monthlycharges'],
                           query_totalcharges = request.form['totalcharges'],
                           query_gender = request.form['gender'],
                           query_partner = request.form['partner'],
                           query_dependents = request.form['dependents'],
                           query_phoneservices = request.form['phoneservice'],
                           query_multiplelines = request.form['multiplelines'],
                           query_internerservice = request.form['internetservice'],
                           query_onlinesecurity = request.form['onlinesecurity'],
                           query_onlinebackup = request.form['onlinebackup'],
                           query_deviceprotection =  request.form['deviceprotection'], 
                           query_techsupport =request.form['techsupport'],
                           query_streamingtv = request.form['streamingtv'],
                           query_streamingmovies = request.form['streamingmovies'],
                           query_contract = request.form['contract'],
                           query_paperlessbilling = request.form['paperlessbilling'],
                           query_paymentmethod = request.form['paymentmethod'],
                           query_tenure= request.form['tenure'])

app.run()
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template
import pickle

app = Flask("__name__",template_folder='template')

dataframe=pd.read_csv('Telco-Customer-Churn_model.csv')

q = ""

@app.route("/")
def loadPage():
	return render_template('home.html', query="")


@app.route("/", methods=['POST'])
def predict():
    
    '''
    SeniorCitizen
    MonthlyCharges
    TotalCharges
    gender
    Partner
    Dependents
    PhoneService
    MultipleLines
    InternetService
    OnlineSecurity
    OnlineBackup
    DeviceProtection
    TechSupport
    StreamingTV
    StreamingMovies
    Contract
    PaperlessBilling
    PaymentMethod
    tenure
    '''
    

    
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    inputQuery8 = request.form['query8']
    inputQuery9 = request.form['query9']
    inputQuery10 = request.form['query10']
    inputQuery11 = request.form['query11']
    inputQuery12 = request.form['query12']
    inputQuery13 = request.form['query13']
    inputQuery14 = request.form['query14']
    inputQuery15 = request.form['query15']
    inputQuery16 = request.form['query16']
    inputQuery17 = request.form['query17']
    inputQuery18 = request.form['query18']
    inputQuery19 = request.form['query19']

    model = pickle.load(open("model.sav", "rb"))
    #model = pickle.load(open("RDForest_model.sav", "rb"))
    
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
             inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,
             inputQuery15, inputQuery16, inputQuery17, inputQuery18, inputQuery19]]
    
    new_df = pd.DataFrame(data, columns = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
                                           'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                           'PaymentMethod', 'tenure'])
    
    df_2 = pd.concat([dataframe, new_df], ignore_index = True) 
    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    
    df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
    #drop column customerID and tenure
    df_2.drop(columns= ['tenure'], axis=1, inplace=True)   
    
    
    
    
    new_df__dummies = pd.get_dummies(df_2[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
           'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
           'Contract', 'PaperlessBilling', 'PaymentMethod','tenure_group']])
    
    
    #final_df=pd.concat([new_df__dummies, new_dummy], axis=1)
        
    
    single = model.predict(new_df__dummies.tail(1))
    probablity = model.predict_proba(new_df__dummies.tail(1))[:,1]
    
    if single==1:
        o1 = "This customer is likely to be churned!!"
        o2 = "Confidence: {}".format(probablity*100)
    else:
        o1 = "This customer is likely to continue!!"
        o2 = "Confidence: {}".format(probablity*100)
        
    return render_template('home.html', output1=o1, output2=o2, 
                           query1 = request.form['query1'], 
                           query2 = request.form['query2'],
                           query3 = request.form['query3'],
                           query4 = request.form['query4'],
                           query5 = request.form['query5'], 
                           query6 = request.form['query6'], 
                           query7 = request.form['query7'], 
                           query8 = request.form['query8'], 
                           query9 = request.form['query9'], 
                           query10 = request.form['query10'], 
                           query11 = request.form['query11'], 
                           query12 = request.form['query12'], 
                           query13 = request.form['query13'], 
                           query14 = request.form['query14'], 
                           query15 = request.form['query15'], 
                           query16 = request.form['query16'], 
                           query17 = request.form['query17'],
                           query18 = request.form['query18'], 
                           query19 = request.form['query19'])
    
app.run()

