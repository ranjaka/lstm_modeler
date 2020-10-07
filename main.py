# imports
import pandas as pd
# import investpy # Not required for now
import matplotlib.pyplot as plt
import numpy as np

#setting figure size (Need to find out how this works exactly)
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

# For normalizing data,
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

# Keras importors
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

import math

# Allows to view coordinates of a point in the plot when hovering
import mplcursors
# --------------------
# Read from CSV
raw_data = pd.read_csv('data/APT Historical Data.csv', usecols=["Date","Price","Open","High"])

# Setting index as dates
raw_data['Date'] = pd.to_datetime(raw_data.Date, format='%b %d, %Y')
raw_data.index = raw_data['Date']

# Sorting data based on date
raw_data_asc = raw_data.sort_index(ascending=True, axis=0)

# Creating new data varaible for using in the processing
model_inputs = pd.DataFrame(index=range(0,len(raw_data_asc)), columns=['Date','Open'])

for i in range(0,len(raw_data_asc)):
    model_inputs['Date'][i] = raw_data_asc['Date'][i]
    model_inputs['Open'][i] = raw_data_asc['Open'][i]

# Setting index
model_inputs.index = model_inputs.Date

# Dropping labels from the columns - this creates a new instancen of the model_inputs object
model_inputs.drop('Date', axis=1, inplace=True)

# creating train and test sets
dataset = model_inputs.values
# length of dataset 1101
lengthDataset = len(dataset)
# Fraction of data to use for training as a percentage (Current 75-25)
lengthTrain = math.floor(0.75*lengthDataset)
# Train and test data spread 75% train and 25% test
train = dataset[0:lengthTrain,:]
print("length of train: ",len(train))
# Test or validation data
valid = dataset[lengthTrain:,:]

print("length of valid: ",len(valid))
# Scalar for normalizing data
scalar = MinMaxScaler(feature_range=(0,1))
# Normalising dataset to between 0 and 1 
scaled_data = scalar.fit_transform(dataset)

# Initialising variables to be used in model training
x_train, y_train = [], []

# predicting remainder of the values using past x points as defined below 
points = len(valid)
# Relationship - single value  of y_train is determined by the  past 60 unit times values of x_train
for i in range (points,len(train)):
    x_train.append(scaled_data[i-points:i,0])
    y_train.append(scaled_data[i,0])

# Rehshaping arrays and prepping for training
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

# create and fit LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=3, batch_size=1, verbose=2)

# validation
inputs = model_inputs[len(model_inputs) - len(valid) - points:].values
inputs = inputs.reshape(-1,1)
inputs = scalar.transform(inputs)

mylist  = []
for i in range(0, inputs.shape[0]):
    mylist.append(inputs[i,0])

predictionPoints = inputs.shape[0]
print("predictionPoints: ",predictionPoints)

# Predicting an extra 14 days beyond the available values
X_test = []
for i in range (points, predictionPoints):
    X_test.append(mylist[i-points:i])  
    X_test_temp = []
    X_test_temp = X_test
    X_test_temp = np.array(X_test_temp)
    X_test_temp = np.reshape(X_test_temp,(X_test_temp.shape[0],X_test_temp.shape[1],1))
    
    opening_price = model.predict(X_test_temp)
    opening_price = opening_price[i-points]
    if(i < predictionPoints):
        mylist[i] = opening_price[0]
    else:
        mylist.append(opening_price[0])



mylist = np.array(mylist)
mylist = np.reshape(mylist,(mylist.shape[0],1))
opening_price = scalar.inverse_transform(mylist)

print("opening price: {0} , size: {1} ".format(opening_price, len(opening_price)))

