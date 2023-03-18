#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import models
from keras import layers
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
df.head(5)


# In[2]:


### Data conversion for network perception

# Useless features

del df['PassengerId']
del df['Name']

df.isnull().sum()


# In[3]:


# Replacing Objects columns with integer  encoding

label_encoder = preprocessing.LabelEncoder()

df['HomePlanet'] = label_encoder.fit_transform(df["HomePlanet"])
df['CryoSleep'] = label_encoder.fit_transform(df["CryoSleep"])
df['Cabin'] = label_encoder.fit_transform(df["Cabin"])
df['Destination'] = label_encoder.fit_transform(df["Destination"])
df['RoomService'] = label_encoder.fit_transform(df["RoomService"])
df['VIP'] = label_encoder.fit_transform(df["VIP"])

# Useless features

df.head(10)


# In[4]:


# Extract Transported feature as target array

y = df['Transported']
del df['Transported']

# Filling NaN in float and int by means

a = df.columns[df.isnull().any()]
for i in a:
    df[i] = df[i].fillna(df[i].mode()[0])  
    
# Convert all to [0 - 1] 

for i in df.columns:
    a = max(df[i])
    b = min(df[i])
    df[i] = (df[i] - b) / a
    
df.head(10)


# In[5]:


# Train/test splitting

x_train, x_test, y_train, y_test = train_test_split(df, y, test_size = 0.2, random_state = 13)

# Build network model

model = keras.Sequential([
    layers.Dense(256, activation="relu"),
    layers.Dropout(.5),
    layers.Dense(128, activation="relu"),
    layers.Dropout(.1),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="Adam",
             loss="binary_crossentropy",
             metrics=["accuracy"])


# In[6]:


# Fit it!

history = model.fit(x_train,
                    y_train,
                    epochs = 35,
                    batch_size = 4,
                    validation_data = (x_test, y_test))

history_dict = history.history
history_dict.keys()


# In[7]:


# Visualization of training process, epoch/loss

history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label = "Training loss")
plt.plot(epochs, val_loss_values, "b", label = "Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Visualization of training process, epoch/accuracy

plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label = "Training acc")
plt.plot(epochs, val_acc, "b", label = "Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[8]:


# Prediction and accuracy

pred = model.predict(x_test)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"test_acc: {test_acc}")


# In[9]:


# Optimal metric of success

from sklearn.metrics import recall_score

p = np.around(pred)
recall_score(y_test, p)


# In[10]:


# Making submission, prepare test data

test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')

del test['PassengerId']
del test['Name']

label_encoder = preprocessing.LabelEncoder()

test['HomePlanet'] = label_encoder.fit_transform(test["HomePlanet"])
test['CryoSleep'] = label_encoder.fit_transform(test["CryoSleep"])
test['Cabin'] = label_encoder.fit_transform(test["Cabin"])
test['Destination'] = label_encoder.fit_transform(test["Destination"])
test['RoomService'] = label_encoder.fit_transform(test["RoomService"])
test['VIP'] = label_encoder.fit_transform(test["VIP"])

a = test.columns[test.isnull().any()]
for i in a:
    test[i] = test[i].fillna(test[i].mode()[0])  

for i in df.columns:
    a = max(test[i])
    b = min(test[i])
    test[i] = (test[i] - b) / a
    
test.head(10)


# In[11]:


# Prediction and transform to True/False

sub = np.round(model.predict(test))
sub = pd.DataFrame(sub)
sub['Transported'] = sub

del sub [0]

sub.loc[sub['Transported'] == 1.0, 'Transported'] = 'True'
sub.loc[sub['Transported'] == 0.0, 'Transported'] = 'False'

sub.head(20)


# In[12]:


# Compile submission file

submission = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
submission = submission['PassengerId']
submission = pd.DataFrame(submission)
submission['Transported'] = sub

submission.to_csv('/kaggle/working/submission.csv', index = False)
submission.head(20)

