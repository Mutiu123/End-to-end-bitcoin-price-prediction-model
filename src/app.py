import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

#load model
model = load_model('C:/Users/adegb/Desktop/Computer Vision Projects/Bitcon-price-prediction/model/BitcoinPricePrediction.keras')
st.header('Bitcoin Price Prediction Model')
st.subheader('10 Years Bitcoin Price data')
data = pd.DataFrame(yf.download('BTC-USD', '2014-06-20', '2024-06-20'))
data = data.reset_index()
st.write(data)

st.subheader('10 Years Bitcoin Line Chart')
# Drop unnecessary columns: 'Date', 'Open', 'High', 'Low', 'Adj Close', and 'Volume'
data.drop(columns=['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
st.line_chart(data)

# Split the data into training and testing data
train_data = data[:-200]
test_data = data[-200:]

# Scale training data into the range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
test_data_scale = scaler.fit_transform(test_data)

# Number of previous days to use for prediction
base_days = 100

# Create input sequences (x) and corresponding target values (y)
x = []
y = []

for i in range(base_days, test_data_scale.shape[0]):
    x.append(test_data_scale[i - base_days:i])
    y.append(test_data_scale[i, 0])
    
# Convert arrays to numpy arrays
x, y = np.array(x), np.array(y)
# Reshape input data to match LSTM input shape
x = np.reshape(x, (x.shape[0],x.shape[1],1))


st.subheader('Predicted vs Original Bitcoin Price')
pred = model.predict(x)
pred = scaler.inverse_transform(pred)
pred = pred.reshape(-1,1)
Orig = scaler.inverse_transform(y.reshape(-1,1))
pred = pd.DataFrame(pred, columns=['Predicted Bitcoin Price'])
Orig = pd.DataFrame(Orig, columns=['Original Price'])
chart_data = pd.concat((pred,Orig), axis=1)
st.write(chart_data)
st.subheader('Predicted vs Original Bitcoin Price Chart')
st.line_chart(chart_data)


#future bitcoin price prediction 
f = y
F= []
future_days = 45
for i in range(base_days, len(f)+future_days):
    f = f.reshape(-1,1)
    inter = [f[-base_days:,0]]
    inter = np.array(inter)
    inter = np.reshape(inter, (inter.shape[0], inter.shape[1],1))
    pred = model.predict(inter)
    f = np.append(f ,pred)
    F = np.append(F, pred)

st.subheader('45 Days Time Bitcoin Price Prediction')   
F = np.array(F)
F =scaler.inverse_transform(F.reshape(-1,1))
st.line_chart(F)
