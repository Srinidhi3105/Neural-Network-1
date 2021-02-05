#forest fires
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import seaborn as sns

#import data 
data = pd.read_csv("C:/Users/SrinidhiR/Desktop/EXELR/assignments/nueral network/forestfires.csv")

#Exploratory data analysis
#checking for NA values
data.isnull().sum()
data.mean()

data.columns
data.nunique()
data[ 'size_category'].unique()
data = data.drop(['month','day'],axis=1)
data.head()
data.shape

#data visualisation
corelation = data.corr()
sns.heatmap(corelation,xticklabels=corelation.columns,yticklabels=corelation.columns,annot =True)

sns.pairplot(data)

sns.relplot(x="temp",y ="area",hue='size_category',data = data)

#splitting data into training and test
train,test = train_test_split(data,test_size=0.2,random_state=45)

#dropping the size_category column from trainX and trainY and adding them to testX,testY
trainX = train.drop(['size_category'],axis =1)
trainY= train['size_category']
testX = test.drop(['size_category'],axis =1)
testY = test['size_category']

#model building
model = Sequential()
model.add(Dense(50,input_dim=3,activation="relu"))
model.add(Dense(40,input_dim=3,activation ="relu"))
model.add(Dense(20,input_dim=3,activation="relu"))
model.add(Dense(1,kernel_initializer= "normal",activation ="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=["accuracy"])

first_model = model

#fitting the array with epoch=10
first_model=first_model.fit(np.array(trainX),np.array(trainY),epochs=100) 

#predicting test data
pred_test = first_model.pred(np.array(trainX))
