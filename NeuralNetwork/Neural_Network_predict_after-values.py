
# coding: utf-8

# In[2]:


from sklearn.model_selection import train_test_split
from pprint import pprint
import pandas as pd
import numpy as np
import sys
import os
import gc
import re
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import RMSprop
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

training_data = pd.read_csv("en_train.csv").sample(n=100000,random_state=40)
batch_size = 128
epochs = 5
max_num_features = 10
space_letter = 0
space_padded_tokens = []
max_data_size = len(training_data)

encoded_classes = pd.factorize(training_data['class'])
# print_data = 100000
x_data = []
labels = encoded_classes[1]
y_data = encoded_classes[0]
gc.collect()
count = 0

for before_value in training_data['before'].values:
    row = np.ones(max_num_features, dtype=int) * space_letter
    for before_value_char, i in zip(list(str(before_value)), np.arange(max_num_features)):
        row[i] = ord(before_value_char)
    count+=1
    x_data.append(row)
    
def make_flat_context_windows(data, pad_size, max_num_features, boundary_letter):
    pad = np.zeros(shape=max_num_features)
    #create array of pad arrays
    pads = [pad for _ in  np.arange(pad_size)]
    data = pads + data + pads
    flattened_context_windows = []
    for lower_bound in np.arange(len(data) - pad_size * 2):
        flattened_context_window = []
        
        context_window_size = pad_size * 2 + 1
        upper_bound = lower_bound + context_window_size
        context_window = data[lower_bound:upper_bound]
        
        for word in context_window:
            flattened_context_window.append([boundary_letter])
            flattened_context_window.append(word)
        flattened_context_window.append([boundary_letter])
        
        flattened_context_windows.append([int(word) for row in flattened_context_window for word in row])
    return flattened_context_windows


x_data = x_data[:max_data_size]
y_data = y_data[:max_data_size]
x_data = np.array(make_flat_context_windows(x_data, pad_size = 1, max_num_features= max_num_features, boundary_letter=-1))
gc.collect()

x_train = np.array(x_data)
y_train = np.array(y_data)
gc.collect()

#adding before and after columns to the data so that it can be used later
b = np.zeros((x_train.shape[0], x_train.shape[1]+1), dtype='O')
b[:,:-1] = x_train
b[:,-1] = np.array(training_data['before'].tolist())

c = np.zeros((x_train.shape[0], 2), dtype='O')
c[:,0] = np.array(y_train.tolist())
c[:,1] = np.array(training_data['after'].tolist())

x_train, x_valid, y_train, y_valid = train_test_split(b, c, test_size=0.2, random_state=2017)

num_classes = len(labels)
y_train1 = keras.utils.to_categorical(y_train[:,:-1], num_classes)
y_valid1 = keras.utils.to_categorical(y_valid[:,:-1], num_classes)
x_train1 = np.reshape(x_train[:,:-1], (x_train[:,:-1].shape[0], x_train[:,:-1].shape[1], 1))
x_valid1 = np.reshape(x_valid[:,:-1], (x_valid[:,:-1].shape[0], x_valid[:,:-1].shape[1], 1))

model = Sequential()
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=((max_num_features * 3) + 4, 1)))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(labels), activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train1, y_train1,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_valid1, y_valid1))


# In[3]:


#Storing all the attributes in one dataframe
a=model.predict(x_train1)
trainDatax=pd.DataFrame(x_train[:,:-1])
trainDatax['predict_class']=pd.DataFrame(np.argmax(a, axis=1).tolist())
trainDatax['before']=pd.DataFrame(x_train[:,-1])

trainDatay=pd.DataFrame(y_train)
trainDatay.columns = ['class','after']

#Change is 0 if before=column else change is 1
trainDatay['change'] = 1
trainDatay['before'] = trainDatax['before']
trainDatay.loc[trainDatay.before == trainDatay.after, 'change'] = 0
trainDatax['predict_class'] = trainDatay['class']

#Splitting data into its respective classes
x_train2 = [0 for x in range(len(labels))]
y_train2 = [0 for x in range(len(labels))]
model1 = [0 for x in range(len(labels))]
model2 = [0 for x in range(len(labels))]
for x in range(len(labels)):
    x_train2[x] = trainDatax[trainDatay['class']==x]
    y_train2[x] = trainDatay[trainDatay['class']==x]
    
    if x_train2[x].shape[0]==0:
        continue
    
    #Input Columns that are needed for predicting change
    list1 = []
    for y in range(x_train2[x].shape[1]-2):
        list1.append(y)
    list1.append('predict_class')
    
    #Training for predicting if change is needed
    model1[x] = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10,10,10,10), activation='tanh',random_state=60)
    model1[x].fit(x_train2[x][list1],y_train2[x][['change']])
    
    y_d = []
    max_num_features=30
    for after_value in y_train2[x]['after'].values:
        row = np.ones(max_num_features, dtype=int) * space_letter
        for after_value_char, i in zip(list(str(after_value)), np.arange(max_num_features)):
            row[i] = ord(after_value_char)
        y_d.append(row)
    
    #if predicted change is 1,then predict the after column
    model2[x] = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(100,100,100,100,100), activation='tanh',random_state=120)
    model2[x].fit(x_train2[x][list1],y_d)


# In[15]:


def testing(X_test,y_test,labels):
    #Predicting class
    x_valid2 = np.reshape(X_test[:,:-1], (X_test[:,:-1].shape[0], X_test[:,:-1].shape[1], 1))
    a=model.predict(x_valid2)
    testDatax=pd.DataFrame(X_test[:,:-1])
    testDatax['predict_class']=pd.DataFrame(np.argmax(a, axis=1).tolist())
    testDatax['before']=pd.DataFrame(X_test[:,-1])
    
    testDatay=pd.DataFrame(y_test)
    testDatay.columns = ['class','after']
    
    #Splitting Data
    x_test1 = [0 for x in range(len(labels))]
    y_test1 = [0 for x in range(len(labels))]
    for x in range(len(labels)):
        print(x)
        x_test1[x] = testDatax[testDatax['predict_class']==x]
        y_test1[x] = testDatay[testDatax['predict_class']==x]

        if x_test1[x].shape[0]==0:
            continue
        
        list1 = []
        for y in range(x_test1[x].shape[1]-2):
            list1.append(y)
        list1.append('predict_class')
        
        #Predicting change
        x_test1[x]['change_predict'] = model1[x].predict(x_test1[x][list1]).astype(int)
        x_test1[x]['predict'] = x_test1[x]['change_predict']
        x_test1[x].loc[x_test1[x].change_predict == 0, 'predict'] = x_test1[x].loc[x_test1[x].change_predict == 0, 'before']
        if x_test1[x][x_test1[x]['change_predict']==1].shape[0]==0:
            continue
        
        #Predicting Output if predicted change is 1
        p = model2[x].predict(x_test1[x][x_test1[x]['change_predict']==1][list1]).astype(int)
        p = np.array(p)
        
        row=0
        for y in x_test1[x][x_test1[x]['change_predict']==1].index.tolist():
            val = ""
            for col in range(p.shape[1]):
                try:
                    if p[row][col]>31 and p[row][col]<58:
                        val = val + chr(p[row][col])
                    elif p[row][col]>64 and p[row][col]<91:
                        val = val + chr(p[row][col])
                    elif p[row][col]>96 and p[row][col]<123:
                        val = val + chr(p[row][col])
                except:
                    continue
            
            x_test1[x].loc[y,'predict'] = val
            row=row+1
        
        
    input1=x_test1[0]
    output1=y_test1[0]
    for x in range(1,len(labels)):
        input1 =pd.concat([input1,x_test1[x]])
        output1 =pd.concat([output1,y_test1[x]])
    
    input1 = input1[['before','predict','predict_class']]
    output1 = output1['after']
    
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    print("Accuracy : "+str(input1[input1['predict']==output1].shape[0]*100/input1.shape[0]))
    print("precision_score : "+str(precision_score(output1, input1['predict'], average="macro")))
    print("recall_score : "+str(recall_score(output1, input1['predict'], average="macro")))
    print("f1_score : "+str(f1_score(output1, input1['predict'], average="macro")))
    return input1,output1


# In[16]:


print("Testing : ")
print("Number of instances : "+str(x_valid.shape[0]))
input2,output2=testing(x_valid,y_valid,labels)


# In[17]:


print("Training : ")
print("Number of instances : "+str(x_train.shape[0]))
input1,output1=testing(x_train,y_train,labels)


# In[26]:


input2['after']=output2
input1['after']=output1

class_transform={}
for x in range(len(labels)):
    class_transform[x]=labels[x]

input1['predict_class']=input1['predict_class'].apply(class_transform.get).astype(str)
input2['predict_class']=input2['predict_class'].apply(class_transform.get).astype(str)

input1.to_csv('pred_train.csv')
input2.to_csv('pred_test.csv')

