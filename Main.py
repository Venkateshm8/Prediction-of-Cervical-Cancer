from tkinter import *
import tkinter
import pandas as pd
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter import ttk
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

main = tkinter.Tk()
main.title("Optimised stacked ensemble techniques in the prediction of cervical cancer using SMOTE and RFERF") #designing main screen
main.geometry("1300x1200")

global filename
global features, X, Y, dataset
global X_train, X_test, y_train, y_test, rfe, clf

def upload():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    dataset = pd.read_csv(filename)
    text.insert(END,'Dataset size  : \n\n')
    text.insert(END,'Total Rows    : '+str(dataset.shape[0])+"\n")
    text.insert(END,'Total Columns : '+str(dataset.shape[1])+"\n\n")
    text.insert(END,'Dataset Samples\n\n')
    text.insert(END,str(dataset.head())+"\n\n")
    text.update_idletasks()
    label = dataset.groupby('Biopsy').size()
    label.plot(kind="bar")
    plt.show()
    
def preprocess():
    global dataset
    text.delete('1.0', END)
    dataset.fillna(0, inplace = True)
    unique, counts = np.unique(dataset['Biopsy'],return_counts=True)
    print(unique)
    print(counts)
    text.insert(END,"Number of Class labels & its count found in dataset before applying SMOTE\n\n") 
    text.insert(END,"Class Label "+str(unique[0])+" found in dataset "+str(counts[0])+"\n")
    text.insert(END,"Class Label "+str(unique[1])+" found in dataset "+str(counts[1])+"\n")

def smoteBalancing():
    global X_train, X_test, y_train, y_test
    global dataset, X, Y
    text.delete('1.0', END)
    Y = dataset.values[:,dataset.shape[1]-1]
    print(Y)
    dataset.drop(['Biopsy'], axis = 1,inplace=True)
    X = dataset.values
    sm = SMOTE(random_state = 42) #creating smote object
    X, Y = sm.fit_sample(X, Y)#applying smote to balance dataset
    unique, counts = np.unique(Y,return_counts=True)
    text.insert(END,"Number of Class labels & its count found in dataset after applying SMOTE\n\n") 
    text.insert(END,"Class Label "+str(unique[0])+" found in dataset "+str(counts[0])+"\n")
    text.insert(END,"Class Label "+str(unique[1])+" found in dataset "+str(counts[1])+"\n\n")
    
    

def featuresSelection():
    global dataset, X, Y, rfe
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    text.insert(END,"Total features found in dataset before applying RFE : "+str(X.shape[1])+"\n")
    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=20) #creating RFE objects
    rfe.fit(X, Y) #applying RFE algorithm to select features
    X = rfe.transform(X)
    text.insert(END,"Total features found in dataset after applying RFE : "+str(X.shape[1])+"\n\n")

    text.insert(END,"Dataset Train & test split is 80% for training and 20% for testing\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Total records used to train machine learning Stacked Ensemble Algorithm is : "+str(X_train.shape[0])+"\n") 
    text.insert(END,"Total records used to test machine learning Stacked Ensemble Algorithm is  : "+str(X_test.shape[0])+"\n")
    

def trainStacked():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, clf
    estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)), ('dt', DecisionTreeClassifier()), ('knn', KNeighborsClassifier(n_neighbors = 2))]
    clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100    
    text.insert(END,'Stacking Ensemble Accuracy  : '+str(a)+"\n")
    text.insert(END,'Stacking Ensemble Precision : '+str(p)+"\n")
    text.insert(END,'Stacking Ensemble Recall    : '+str(r)+"\n")
    text.insert(END,'Stacking Ensemble FScore    : '+str(f)+"\n\n")
    text.update_idletasks()
    LABELS = ['Normal','Cervical Cancer'] 
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title("Stacking Ensemble Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    

def predict():
    text.delete('1.0', END)
    global clf, rfe
    testfile = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(testfile)
    dataset.fillna(0, inplace = True)
    dataset = dataset.values
    dataset = rfe.transform(dataset)
    print(dataset.shape)
    predict = clf.predict(dataset)
    print(predict)
    for i in range(len(predict)):
        if predict[i] == 0:
            text.insert(END,"TEST DATA = "+str(dataset[i])+" =====> PREDICTED AS NORMAL\n\n")
        if predict[i] == 1:
            text.insert(END,"TEST DATA = "+str(dataset[i])+" =====> PREDICTED AS CERVICAL CANCER\n\n")    

def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='Optimised stacked ensemble techniques in the prediction of cervical cancer using SMOTE and RFERF')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=100)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Cervical Cancer Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=preprocess)
processButton.place(x=50,y=150)
processButton.config(font=font1) 

smoteButton = Button(main, text="Data Balancing using SMOTE", command=smoteBalancing)
smoteButton.place(x=50,y=200)
smoteButton.config(font=font1) 

featuresButton = Button(main, text="Features Selection using RFERF", command=featuresSelection)
featuresButton.place(x=50,y=250)
featuresButton.config(font=font1) 

stackedButton = Button(main, text="Trained Stacked Ensemble Algorithm", command=trainStacked)
stackedButton.place(x=50,y=300)
stackedButton.config(font=font1)

predictButton = Button(main, text="Predict Cancer from Test Data", command=predict)
predictButton.place(x=50,y=350)
predictButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50,y=400)
exitButton.config(font=font1)


main.config(bg='OliveDrab2')
main.mainloop()
