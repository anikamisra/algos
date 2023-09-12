import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

# ------------------------------------------------------------------------------------------------------------
# PREPARE DATA 
# ------------------------------------------------------------------------------------------------------------
dataset = pd.read_csv("Name gender dataset - name_gender_dataset.csv")
dataset.head()
del dataset['Probability']
del dataset['Count']

print(dataset.shape) 
# There are 147,268 rows and 2 columns. Meaning there are 147,268 names. Let's see how many of those names are repeated.

print(len(dataset['Name'].unique()))
# 13359 repeated names 

# ------------------------------------------------------------------------------------------------------------
# VISUALIZATIONS
# ------------------------------------------------------------------------------------------------------------
# visualizing class imbalance 
sns.countplot(x='Gender', data = dataset)
plt.title('No. of male and female names in the dataset')
plt.xticks([0,1],('Female','Male'))

# visualizing letter counts 
alphabets = ['A', 'B', 'C', 'D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
startletter_count = {}
for i in alphabets:
  startletter_count[i] = len(dataset[dataset['Name'].str.startswith(i)])
print(startletter_count)
plt.figure(figsize = (16,8))
plt.bar(startletter_count.keys(), startletter_count.values())
plt.xlabel('Letter that the name starts with')
plt.ylabel('No. of names')


# ------------------------------------------------------------------------------------------------------------
# PREPARING FOR MODEL 
# ------------------------------------------------------------------------------------------------------------
# predicting gender from name! 
X = list(dataset['Name'])
Y = list(dataset['Gender'])

#now we will convert the F and M labels to machine-readable format
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
Y = encoder.fit_transform(Y)

# count vectorize names with character level n-grams as features 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(analyzer='char')
X=cv.fit_transform(X).toarray()

# randomize data and split into testing and training sets 
from sklearn.model_selection import train_test_split 
dataset = dataset.sample(n = 147268, random_state = 0)
dataset = dataset.drop([15041])
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42) 

# ------------------------------------------------------------------------------------------------------------
# MODEL 
# ------------------------------------------------------------------------------------------------------------
# logistic regression 
from sklearn.linear_model import LogisticRegression  
LR_model = LogisticRegression(max_iter=1000)
LR_model.fit(x_train, y_train)
LR_y_pred = LR_model.predict(x_test)

# Naive Bayes 
from sklearn.naive_bayes import MultinomialNB
NB_model = MultinomialNB()
NB_model.fit(x_train, y_train)
NB_y_pred = NB_model.predict(x_test)

# XGBoost 
from xgboost import XGBClassifier
# depending on which version of python u have u might not need the parameter 
XGB_model = XGBClassifier(use_label_encoder = False)
XGB_model.fit(x_train,y_train)
XGB_y_pred = XGB_model.predict(x_test)

# ------------------------------------------------------------------------------------------------------------
# MODEL ANALYTICS 
# ------------------------------------------------------------------------------------------------------------
# On the blog, it said XGBoost was the most accurate, but our results might be slightly different because our dataset is different. So, let's take a look.
from sklearn.metrics import confusion_matrix
def cmatrix(model):
  y_pred = model.predict(x_test)
  cmatrix = confusion_matrix(y_test, y_pred)
  print(cmatrix)

  sns.heatmap(cmatrix, fmt='d', cmap='BuPu', annot=True)
  plt.xlabel('Predicted Values')
  plt.ylabel('Actual Values')
  plt.title('Confusion Matrix')

# logistic regression model analytics 
import sklearn.metrics as metrics
print(metrics.accuracy_score(LR_y_pred, y_test))
print(metrics.classification_report(y_test, LR_y_pred))
print(cmatrix(LR_model))

# naive bayes model analytics 
print(metrics.accuracy_score(NB_y_pred, y_test))
print(metrics.classification_report(y_test, NB_y_pred))
print(cmatrix(NB_model))

# XGBoost model analytics 
print(metrics.accuracy_score(XGB_y_pred, y_test))
print(metrics.classification_report(y_test, XGB_y_pred))
print(cmatrix(XGB_model))

# ------------------------------------------------------------------------------------------------------------
# LSTM MODEL  
# ------------------------------------------------------------------------------------------------------------
from tensorflow.keras import models
from tensorflow.keras.models import Model 
from tensorflow.keras.models import load_model 
from keras.layers import Embedding 
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Dense, Dropout
from tensorflow.keras.layers import LSTM 

#defining the LSTM layers 
max_words = 1000
voc_size = 1000
max_len = 26 
LSTM_model = Sequential()
LSTM_model.add(Embedding(voc_size, 40, input_length=50))
LSTM_model.add(Dropout(0.3))
LSTM_model.add(LSTM(100))
LSTM_model.add(Dropout(0.3))
LSTM_model.add(Dense(64, activation = 'relu'))
LSTM_model.add(Dropout(0.3))
LSTM_model.add(Dense(1,activation='sigmoid'))
LSTM_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(LSTM_model.summary())
LSTM_model.fit(x_train,y_train, epochs=100, batch_size=64) 

# ------------------------------------------------------------------------------------------------------------
# TEST YOUR RESULTS   
# ------------------------------------------------------------------------------------------------------------
# change this line of code to test a new name! 
new_name = "Anika"
# do not change past this line 
new_name_vector = cv.transform([new_name])
prediction = LSTM_model.predict(new_name_vector)
threshold = 0.44
prediction_binary = np.where(prediction >= threshold, "boy", "girl")
print(prediction_binary)
