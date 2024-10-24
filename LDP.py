from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from imutils import paths
from tkinter.filedialog import askopenfilename
from tkinter import scrolledtext

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
#get_ipython().run_line_magic('matplotlib', 'inline')
import scikitplot as skplt
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network  import MLPClassifier

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

import os
from os import path

main = tkinter.Tk()
main.title("Severity of Liver Fibrosis for Chronic HBV based on Physical Layer with Serum Markers")
main.geometry("1300x1200")

global filename
global raw_data
global X, y, X_train, X_test, y_train, y_test
global ltsm_acc, ann_acc, mlp_acc,cnn_acc
global MODEL_PATH


def upload():
    global filename
    text.delete('1.0', END)
    filename = askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    text.insert(END, "Dataset loaded\n\n")

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

def preprocess():
    global filename
    global raw_data
    global X,y
    
    text.delete('1.0',END)

    text.insert(END,"Importing dataset\n")
    raw_data = pd.read_excel(filename)
    
    text.insert(END,"Data column information: "+str(raw_data.columns)+"\n\n")
    text.insert(END,"data shape"+str(raw_data.shape)+"\n\n")

    raw_data = raw_data.drop(['Physical Activity','PVD', 'Source of Care','Family  HyperTension','Family Hepatitis','Chronic Fatigue','PVD','Region'],axis=1)
    raw_data.head()

    raw_data['Gender'] = raw_data['Gender'].map({'F': 0, 'M': 1})


    text.insert(END,"Top Absolute Correlations")
    text.insert(END,"Top Correlation values: "+str(get_top_abs_correlations(raw_data, 10))+"\n\n")

    raw_data.isnull().sum()

    raw_data = raw_data.drop(['Weight','Obesity', 'Waist','Bad Cholesterol'],axis=1)

    raw_data.dtypes

    print("Top Absolute Correlations")
    print(get_top_abs_correlations(raw_data, 10))

    raw_data.isnull().sum()

    cols_mode = ['Hepatitis', 'Diabetes', 'HyperTension', 'Education', 'Unmarried','PoorVision','Income']
    for column in cols_mode:
        raw_data[column].fillna(raw_data[column].mode()[0], inplace=True)

    cols_mode = ['Height', 'Body Mass Index', 'Maximum Blood Pressure', 'Minimum Blood Pressure', 'Good Cholesterol','Total Cholesterol','Income']
    for column in cols_mode:
        raw_data[column].fillna(raw_data[column].mean(), inplace=True)

    raw_data.isnull().sum()

    raw_data.dtypes

    y = raw_data['ALF']
    raw_data.drop(columns=['ALF'],inplace=True)

    X = raw_data
    y = y[:6000]

def dataSplit():

    global X,y
    global X_train,X_test,y_train,y_test
    text.delete('1.0',END)
    
    scaler = MinMaxScaler()
    scaler.fit(X)

    X = pd.DataFrame(scaler.transform(X),columns=X.columns)
    X.head()
    X_pred = X[:6000]
    X = X[:6000]

    X_train, X_test, y_train, y_test = train_test_split(X,y,stratify = y,shuffle=True ,test_size=0.2)

    text.insert(END,"Spliting the data is done")


def logit():
    global X_train, X_test, y_train, y_test

    text.delete('1.0',END)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    text.insert(END,"Score of logistic Algo: "+str(lr.score(X_test, y_test))+"\n\n")
    
    y_pred = lr.predict(X_test)

    text.insert(END,"Classification Report: "+str(classification_report(y_test, y_pred))+"\n\n")

    text.insert(END,"Confusion Matrix : "+str(confusion_matrix(y_test,y_pred))+"\n\n")

def xgb():

    global X_train, X_test, y_train, y_test

    text.delete('1.0',END)

    xgb = XGBClassifier(random_state=10)

    xgb.fit(X_train,y_train)

    text.insert(END,"Score of XGB Algo: "+str(xgb.score(X_test, y_test))+"\n\n")

    y_pred = xgb.predict(X_test)
    
    text.insert(END,"Classification Report: "+str(classification_report(y_test, y_pred))+"\n\n")

    text.insert(END,"Confusion Matrix : "+str(confusion_matrix(y_test,y_pred))+"\n\n")


def knn():

    global X_train, X_test, y_train, y_test

    text.delete('1.0',END)

    k_range = range(1,15)
    scores = {}
    scores_list = []
    for k in k_range:
        knn =  KNeighborsClassifier(n_neighbors = k)
        knn.fit(X_train, y_train)
        y_predict = knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test, y_predict)
        scores_list.append(metrics.accuracy_score(y_test, y_predict))
        text.insert(END,"Score of KNN : "+str(knn.score(X_test, y_test))+"  K value : "+str(k)+"\n\n")


    knn =  KNeighborsClassifier(n_neighbors = 5)
    knn.fit(X_train, y_train)

    text.insert(END,"Score of KNN Algo: "+str(knn.score(X_test, y_test))+"\n\n")
    y_pred = knn.predict(X_test)
    text.insert(END,"Classification Report: "+str(classification_report(y_test, y_pred))+"\n\n")

    text.insert(END,"Confusion Matrix : "+str(confusion_matrix(y_test,y_pred))+"\n\n")

def dt():

    global X_train, X_test, y_train, y_test

    text.delete('1.0',END)

    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train,y_train)

    text.insert(END,"Score of DT Algo: "+str(dt.score(X_test, y_test))+"\n\n")
    y_pred = dt.predict(X_test)
    text.insert(END,"Classification Report: "+str(classification_report(y_test, y_pred))+"\n\n")

    text.insert(END,"Confusion Matrix : "+str(confusion_matrix(y_test,y_pred))+"\n\n")

def rf():

    global X_train, X_test, y_train, y_test

    text.delete('1.0',END)

    rf = RandomForestClassifier(n_estimators = 10,max_depth=2, random_state=0)
    rf.fit(X_train, y_train)

    text.insert(END,"Score of RF Algo: "+str(rf.score(X_test, y_test))+"\n\n")
    y_pred = rf.predict(X_test)
    text.insert(END,"Classification Report: "+str(classification_report(y_test, y_pred))+"\n\n")

    text.insert(END,"Confusion Matrix : "+str(confusion_matrix(y_test,y_pred))+"\n\n")

def adc():

    global X_train, X_test, y_train, y_test

    text.delete('1.0',END)

    adc = AdaBoostClassifier(n_estimators=100, random_state=0)
    adc.fit(X_train, y_train)

    text.insert(END,"Score of ADC Algo: "+str(adc.score(X_test, y_test))+"\n\n")

    adcx = AdaBoostClassifier(n_estimators=100, random_state=0,base_estimator=XGBClassifier(random_state=10))
    adcx.fit(X_train, y_train)
    text.insert(END,"Score of ADC XGB Algo: "+str(adcx.score(X_test, y_test))+"\n\n")

    adcs = AdaBoostClassifier(n_estimators=100, random_state=0,base_estimator=SVC(),algorithm='SAMME')
    adcs.fit(X_train, y_train)
    text.insert(END,"Score of ADC+SVC Algo: "+str(adcs.score(X_test, y_test))+"\n\n")

    adcl = AdaBoostClassifier(n_estimators=100, random_state=0,base_estimator=LogisticRegression())
    adcl.fit(X_train, y_train)
    text.insert(END,"Score of ADC+Logit Algo: "+str(adcl.score(X_test, y_test))+"\n\n")

    y_pred = adcl.predict(X_test)
    text.insert(END,"Classification Report: "+str(classification_report(y_test, y_pred))+"\n\n")

    text.insert(END,"Confusion Matrix : "+str(confusion_matrix(y_test,y_pred))+"\n\n")


def svc():
    global X_train, X_test, y_train, y_test

    text.delete('1.0',END)

    svmg = SVC(gamma= 0.0000001, C=0.2,max_iter=100,probability=True)
    svmg.fit(X_train, y_train)

    text.insert(END,"Score of SVC Algo: "+str(svmg.score(X_test, y_test))+"\n\n")

    y_pred = svmg.predict(X_test)
    text.insert(END,"Classification Report: "+str(classification_report(y_test, y_pred))+"\n\n")

    text.insert(END,"Confusion Matrix : "+str(confusion_matrix(y_test,y_pred))+"\n\n")


def hgb():
    global X_train, X_test, y_train, y_test

    text.delete('1.0',END)

    hgb = HistGradientBoostingClassifier().fit(X_train, y_train)
    text.insert(END,"Score of HGB Algo: "+str(hgb.score(X_test, y_test))+"\n\n")

    y_pred = hgb.predict(X_test)
    text.insert(END,"Classification Report: "+str(classification_report(y_test, y_pred))+"\n\n")

    text.insert(END,"Confusion Matrix : "+str(confusion_matrix(y_test,y_pred))+"\n\n")

def stackclassify():
    global X_train, X_test, y_train, y_test
    text.delete('1.0',END)

    estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
              ('svr', make_pipeline(LinearSVC(random_state=42)))]
    sc = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    sc.fit(X_train, y_train).score(X_test, y_test)

    text.insert(END,"Score of stackclassify Algo: "+str(sc.score(X_test, y_test))+"\n\n")

    y_pred = sc.predict(X_test)
    text.insert(END,"Classification Report: "+str(classification_report(y_test, y_pred))+"\n\n")

    text.insert(END,"Confusion Matrix : "+str(confusion_matrix(y_test,y_pred))+"\n\n")

def mlp():

    global X_train, X_test, y_train, y_test
    text.delete('1.0',END)

    mlp = MLPClassifier(activation='tanh',solver='sgd',learning_rate='adaptive')
    mlp.fit(X_train,y_train)
    mlp.score(X_test,y_test)


    mlp = MLPClassifier(activation='logistic',solver='sgd',learning_rate='adaptive')
    mlp.fit(X_train,y_train)
    mlp.score(X_test,y_test)

    text.insert(END,"Score of MLP Algo: "+str(mlp.score(X_test, y_test))+"\n\n")

    y_pred = mlp.predict(X_test)
    text.insert(END,"Classification Report: "+str(classification_report(y_test, y_pred))+"\n\n")

    text.insert(END,"Confusion Matrix : "+str(confusion_matrix(y_test,y_pred))+"\n\n")

def ann():
    global X_train, X_test, y_train, y_test
    text.delete('1.0',END)

    #input and output layer is of 20 and 4 dimensions respectively.
    #Dependencies
    # Neural network

    if (path.exists("model_ann.h5")):
        # load model
        model = load_model('model_ann.h5')
    else:
        model = Sequential()
        model.add(Dense(16, input_dim=18, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=100, batch_size=5)

        model.save("model_ann.h5")

    # summarize model.
    text.insert(END,"ANN Model summary: \n"+str(model.summary())+"\n\n")

    _, accuracy = model.evaluate(X_test, y_test,verbose=0)
    text.insert(END,'Accuracy: '+str(accuracy*100)+"\n\n")

font = ('times', 16, 'bold')
title = Label(main, text='Comparitive Analysis of Liver Disease Prediction')
title.config(bg='PaleGreen2', fg='Khaki4')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0, y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Dataset", command=upload)
upload.place(x=700, y=100)
upload.config(font=font1)

pathlabel=Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=700, y=150)

preprocess = Button(main, text="Preprocess Dataset", command=preprocess)
preprocess.place(x=700, y=200)
preprocess.config(font=font1)

model = Button(main, text="Generate Train and Test data for Model", command=dataSplit)
model.place(x=700, y=250)
model.config(font=font1)

runann = Button(main, text="Run Logistic Algorithm", command=logit)
runann.place(x=700, y=300)
runann.config(font=font1)

runltsm = Button(main, text="Run KNN Algorithm", command=knn)
runltsm.place(x=700, y=350)
runltsm.config(font=font1)

runcnn = Button(main, text="Run RF Algorithm", command=rf)
runcnn.place(x=700, y=400)
runcnn.config(font=font1)

runmlp = Button(main, text="Run DT Algorithm", command=dt)
runmlp.place(x=700, y=450)
runmlp.config(font=font1)

runann = Button(main, text="Run SVC Algorithm", command=svc)
runann.place(x=700, y=500)
runann.config(font=font1)

runltsm = Button(main, text="Run ADC Algorithm", command=adc)
runltsm.place(x=700, y=550)
runltsm.config(font=font1)

runcnn = Button(main, text="Run XGB Algorithm", command=xgb)
runcnn.place(x=700, y=600)
runcnn.config(font=font1)

runmlp = Button(main, text="Run HGB Algorithm", command=hgb)
runmlp.place(x=700, y=650)
runmlp.config(font=font1)

runltsm = Button(main, text="Run MLP Algorithm", command=mlp)
runltsm.place(x=700, y=700)
runltsm.config(font=font1)

runcnn = Button(main, text="Run ANN Algorithm", command=ann)
runcnn.place(x=700, y=750)
runcnn.config(font=font1)

runmlp = Button(main, text="Run stackclassify Algorithm", command=stackclassify)
runmlp.place(x=700, y=800)
runmlp.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='PeachPuff2')
main.mainloop()
    
