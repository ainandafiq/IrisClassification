from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import PySimpleGUI as sg

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def isfloat(number) :
    try:
        float(number)
        return True
    except :
        return False

def showclass(flower) :
    if(flower==0):
        return "Iris Setosa"
    elif(flower==1):
        return "Iris Versicolor"
    elif(flower==2):
        return "Iris Virginica"

TRAIN_PATH = "train.json"
TEST_PATH = "test.json"

f_train = open(TRAIN_PATH)
f_test = open(TEST_PATH)

train_data = json.load(f_train)
test_data = json.load(f_test)

x_train = np.array(train_data["features"])
y_train = np.array(train_data["label"])

x_test = np.array(test_data["features"])
y_test = np.array(test_data["label"])

gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
accuracyNB = metrics.accuracy_score(y_test, y_pred)
print('[[prediction Gaussian NB]]')
print('prediction: ', y_pred)
print('actual: ', y_test)
print('accuracy: ', accuracyNB)
print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != y_pred).sum()))

sg.theme('DarkAmber')   # Add a touch of color
# All the stuff inside your window.
layout = [  [sg.Text('Masukkan nilai (cm)')],
            [sg.Text('Panjang sepal bunga'), sg.InputText()],
            [sg.Text('Lebar sepal bunga'), sg.InputText()],
            [sg.Text('Panjang kelopak bunga'), sg.InputText()],
            [sg.Text('Lebar kelopak bunga'), sg.InputText()],
            [sg.Button('Ok'), sg.Button('Close Window')]
         ]

window = sg.Window('Prediksi Bunga Iris', layout).Finalize()
while True:
    event, values = window.read()
    if event in (None, 'Close Window'): # if user closes window or clicks cancel
        break
    sl = values[0]
    sw = values[1]
    pl = values[2]
    pw = values[3]
    list_val_is_num = [
        isfloat(sl),
        isfloat(sw),
        isfloat(pl),
        isfloat(pw)
    ]
    if not(all(list_val_is_num)) : sg.popup('Mohon isikan semua kolom dengan angka')
    else :

        list_val = [sl,sw,pl,pw]
        np_val = np.array(list_val, dtype=float)
        pred = gnb.predict(np_val.reshape(1, -1))
        sg.popup('Komputer memprediksi bahwa bunga tersebut adalah : ', showclass(pred))

window.close()
