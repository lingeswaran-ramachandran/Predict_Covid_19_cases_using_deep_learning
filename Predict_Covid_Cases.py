# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 09:03:57 2022

@author: User
"""
import os
import pickle
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
from modules_predict_covid_cases import EDA

#%% STATICS
CSV_PATH = os.path.join(os.getcwd(),'cases_malaysia_train.csv')

log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
COVID_LOG_FOLDER_PATH = os.path.join(os.getcwd(),'covid_log',log_dir)
COVID_MODEL_SAVE_PATH = os.path.join(os.getcwd(),'covid_model.h5')

#EDA
# Step 1) Data loading
df = pd.read_csv(CSV_PATH)
df.info() # check for Null
temp = df.describe().T # percentile, mean, min-max, count

# Step 2) Data Inspection
eda = EDA() 
eda.plot_graph(df) # to plot the graphs

# Step 3) Data Cleaning
# Data Imputation
df['cases_new'] = pd.to_numeric(df['cases_new'],errors='coerce') # to convert object to float
df.info()
df.isna().sum() # to check for duplicated data

# use interpolate for NaNs value
df['cases_new'].interpolate(method='polynomial',order=2,inplace=True) # to fill NaN for timeseries data
df.isna().sum()

temp = df['cases_new']

# Step 4) Feature selection
# we are now selecting cases_new data only

# Step 5) Pre-processing
mms = MinMaxScaler() # to initiate MinMaxScaler
df = mms.fit_transform(np.expand_dims(df['cases_new'],axis=-1))

 # save pickle
MMS_FILE_NAME = os.path.join(os.getcwd(),'covid_cases_scalar.pkl')
with open(MMS_FILE_NAME,'wb') as file:
    pickle.dump(mms,file)

X_train = []
y_train = []

win_size = 30 # use last 30 days of number of cases
 
# i = 30
for i in range(win_size,np.shape(df)[0]): #df.shape[0]
    X_train.append(df[i-win_size:i,0])
    y_train.append(df[i,0]) 
 
X_train = np.array(X_train) # to expand the dimension
y_train = np.array(y_train) # to expand the dimension

#%%
model = Sequential()
model.add(Input(shape=(np.shape(X_train)[1],1))) # input_length, # number of features
model.add(LSTM(64,return_sequences=((True)))) # LSTM
model.add(Dropout(0.3))
model.add(LSTM(64)) # LSTM
model.add(Dropout(0.3))
model.add(Dense(1,activation='relu')) # Output Layer
model.summary()

model.compile(optimizer='adam',loss='mse',metrics='mape')

# callbacks
tensorboard_callback = TensorBoard(log_dir=COVID_LOG_FOLDER_PATH)

early_stopping_callback = EarlyStopping(monitor='loss',patience=3)

X_train = np.expand_dims(X_train,axis=-1)
hist = model.fit(X_train,y_train,batch_size=32,epochs=100,
                 callbacks=[tensorboard_callback,early_stopping_callback])

#%% model saving
model.save(COVID_MODEL_SAVE_PATH)

#%%

plot_model(model,show_layer_names=(True),show_shapes=(True))
#%% model evaluation
hist.history.keys()

plt.figure()
plt.plot(hist.history['mape'])
plt.show()

plt.figure()
plt.plot(hist.history['loss'])
plt.show()

#%% model development and analysis
CSV_TEST_PATH = os.path.join(os.getcwd(),'cases_malaysia_test.csv')

test_df = pd.read_csv(CSV_TEST_PATH)
test_df.info()

test_df['cases_new']=pd.to_numeric(test_df['cases_new'],errors='coerce') # to convert object to float
test_df.info() # got 1 Nans

# use interpolate for NaNs value
test_df['cases_new'].interpolate(method='polynomial',order=2,inplace=True) # to fill NaN for timeseries data
test_df.isna().sum() # 0 Nans

test_df = mms.transform(np.expand_dims(test_df.iloc[:,1],axis=-1)) # select the first column of test_df and do scaling
con_test = np.concatenate((df,test_df),axis=0) # # to concanate Open data from train and Open data from test
con_test = con_test[-(win_size+len(test_df)):]

X_test = []
for i in range(win_size,len(con_test)):
    X_test.append(con_test[i-win_size:i,0])

X_test = np.array(X_test)

predicted = model.predict(np.expand_dims(X_test,axis=-1))

#%% ploting the graph

plt.figure()
plt.plot(test_df,'b',label='actual covid cases')
plt.plot(predicted,'r',label='predicted covid cases')
plt.legend()
plt.show()

plt.figure()
plt.plot(mms.inverse_transform(test_df),'b',label='actual covid cases')
plt.plot(mms.inverse_transform(predicted),'r',label='predicted covid cases')
plt.legend()
plt.show()

#%% MSE, MAPE

print(mean_absolute_error(test_df, predicted))
print(mean_squared_error(test_df, predicted))

test_df_inverse = mms.inverse_transform(test_df) # 
predicted_inverse = mms.inverse_transform(predicted)

print(mean_absolute_error(test_df_inverse,predicted_inverse))
print(mean_squared_error(test_df_inverse,predicted_inverse))
print(mean_absolute_percentage_error(test_df_inverse,predicted_inverse))
print((mean_absolute_error(test_df, predicted)/sum(abs(test_df))) *100)

#%% Discussion

# The model is able to predict the trend of the covid-19 cases
# 21 % MAPE when tested against testingt dataset
# The graph is overfitting 
# Can increase the dropout rate and neuron to increase the performance

