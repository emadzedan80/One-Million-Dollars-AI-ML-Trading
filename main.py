# IMPORTING IMPORTANT LIBRARIES

import tkinter as tk
import webbrowser as wb
from tkinter import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import mplfinance as mpf

# FUNCTION TO CREATE 1D DATA INTO TIME SERIES DATASET
def new_dataset(dataset, step_size):
	data_X, data_Y = [], []
	for i in range(len(dataset) - step_size-1):
		a = dataset[i:(i + step_size), 0]
		data_X.append(a)
		data_Y.append(dataset[i + step_size, 0])
	return np.array(data_X), np.array(data_Y)

# THIS FUNCTION CAN BE USED TO CREATE A TIME SERIES DATASET FROM ANY 1D ARRAY

window_open = False
if window_open == False:
    window_open = True
    window = tk.Tk()
    window.title("One Million Dollars Artificial Intelligence & Machine Learning Trading")
    window.config(padx=10, pady=10)
    window.geometry("700x460")
    window.resizable(False, False)
    window.eval('tk::PlaceWindow . center')

def on_closing():
    window.destroy()

def openwebsite( url ):
   wb.open_new_tab(url)

yahoolink = Button(text="https://finance.yahoo.com/currencies", command=lambda: openwebsite("https://finance.yahoo.com/currencies"))
yahoolink.pack(pady=5)
symbol30dayslabel = Label(text="One Million Dollars\nArtificial Intelligence & Machine Learning Trading!", font=("Arial", 20))
symbol30dayslabel.pack(pady=5)
entersymbollabel = Label(text="Enter Symbol:")
entersymbollabel.pack(pady=5)
symbolentery = Entry(justify='center')
symbolentery.pack(pady=5)
looplabel = Label(text="Enter Number Of Loops:\n(For First Time Keep It 1 To Get The Symbol Number Fast)\nFor Best Results Set The Loops Number To 50 (Every loop Takes About 15 Seconds)\n(Analysis For 50 Might Take 20 Minutes.)")
looplabel.pack(pady=5)
loopentery = Entry(justify='center')
loopentery.insert(END, '1')
loopentery.pack(pady=5)
entererrorlabel = Label(text="Enter Symbol Number (When You have It):\n(For Now Keep It 1)")
entererrorlabel.pack(pady=5)
errorentery = Entry(justify='center')
errorentery.insert(END, '1')
errorentery.pack(pady=5)

text = tk.StringVar()
text.set("Symbol Number Will Show Here...")

resultlabel = Label(window, textvariable=text, bg='#5eba7d')
resultlabel.pack(pady=5)

def setsymbol(symbol, errornumber):
    stock_symbol = symbol

    # 1d timeframe for 10 year
    df = yf.download(tickers=stock_symbol, period='3660d', interval='1d')
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df[['Date', 'Open', 'High', 'Low', 'Close']]
    df = df.drop_duplicates(keep='first')
    df.to_csv('Data/Data.csv', index=True, index_label='Date')
    # mpf.plot(df, style='yahoo', title='AUDUSD Chart:', type='hollow_and_filled', savefig='Output/AUDUSD_OldChart.png')
    mpf.plot(df, style='yahoo', type='hollow_and_filled', title=symbolentery.get() + ' Chart:')

    # IMPORTING DATASET
    dataset = pd.read_csv('Data/Data.csv', usecols=['Date', 'Open', 'High', 'Low', 'Close'])
    dataset = dataset.reindex(index=dataset.index[::-1])

    # print(len(dataset))
    # print(dataset.tail())
    # print(dataset.head())

    opn = dataset[['Open']]
    ds = opn.values

    # Using MinMaxScaler for normalizing data between 0 & 1
    normalizer = MinMaxScaler(feature_range=(0, 1))
    ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1, 1))

    # Defining test and train data sizes
    train_size = int(len(ds_scaled) * 0.70)
    test_size = len(ds_scaled) - train_size

    # Splitting data between train and test
    ds_train, ds_test = ds_scaled[0:train_size, :], ds_scaled[train_size:len(ds_scaled), :1]

    # creating dataset in time series for LSTM model
    # X[100,120,140,160,180] : Y[200]
    def create_ds(dataset, step):
        x_axis_train, y_axis_train = [], []
        for j in range(len(dataset) - step - 1):
            a = dataset[j:(j + step), 0]
            x_axis_train.append(a)
            y_axis_train.append(dataset[j + step, 0])
        return np.array(x_axis_train), np.array(y_axis_train)

    # Taking 100 days price as one record for training
    time_stamp = 100
    X_train, y_train = create_ds(ds_train, time_stamp)
    X_test, y_test = create_ds(ds_test, time_stamp)

    # Reshaping data to fit into LSTM model
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Creating LSTM model using keras
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(16, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1, activation='linear'))
    model.summary()

    # Training model with adam optimizer and mean squared error loss function
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=int(loopentery.get()), batch_size=64)

    # Plotting loss, it shows that loss has decreased significantly and model trained well
    # modle performance visualization
    # "Loss"
    # plt.plot(model.history.history['loss'], color='#4277ff')
    # plt.ylabel("Loss")
    # plt.xlabel("Value")
    # plt.title("Model Loss: it shows that loss has decreased significantly and model trained well")
    # plt.show()

    # Predicitng on train and test data
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform to get actual value
    train_predict = normalizer.inverse_transform(train_predict)
    test_predict = normalizer.inverse_transform(test_predict)

    # Getting the last 100 days records
    fut_inp = ds_test[270:]
    fut_inp = fut_inp.reshape(1, -1)
    tmp_inp = list(fut_inp)

    # Creating list of the last 100 data
    tmp_inp = tmp_inp[0].tolist()

    # Predicting next 30 days price using the current data
    # It will predict in sliding window manner (algorithm) with stride 1
    lst_output = []
    # Take this number from the error message
    try:
        n_steps = int(errornumber)
        i = 0
        while i < 2:

            if len(tmp_inp) > 100:
                fut_inp = np.array(tmp_inp[1:])
                fut_inp = fut_inp.reshape(1, -1)
                fut_inp = fut_inp.reshape((1, n_steps, 1))
                yhat = model.predict(fut_inp, verbose=0)
                tmp_inp.extend(yhat[0].tolist())
                tmp_inp = tmp_inp[1:]
                lst_output.extend(yhat.tolist())
                i = i + 1
            else:
                fut_inp = fut_inp.reshape((1, n_steps, 1))
                yhat = model.predict(fut_inp, verbose=0)
                tmp_inp.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i = i + 1

        # print(lst_output)
        ds_new = ds_scaled.tolist()
        # Exntends helps us to fill the missing value with approx value
        ds_new.extend(lst_output)

        # Creating final data for plotting
        final_graph = normalizer.inverse_transform(ds_new)

        #plt.plot(final_graph)
        #plt.title(stock_symbol + " Prediction Of Next 30 Days")
        #plt.axhline(y=final_graph[len(final_graph) - 1], color='#4277ff', linestyle=':', label='NEXT 30 Days: {0}'.format(round(float(*final_graph[len(final_graph) - 1]), 2)))
        #plt.show()

        my_dataframe = pd.DataFrame(final_graph.reshape(-1,5), columns=['Date', 'Open', 'High', 'Low', 'Close'])
        my_dataframe.index = pd.DatetimeIndex(my_dataframe['Date'])
        my_dataframe = my_dataframe.drop_duplicates(keep='first')
        #mpf.plot(my_dataframe, style='yahoo', title='AUDUSD Prediction:', type='hollow_and_filled', savefig='Output/AUDUSD_Prediction.png')
        mpf.plot(my_dataframe, style='yahoo', type='hollow_and_filled', title=stock_symbol + ' Prediction:')
    except Exception as e:
        print(str(e))
        sliced = str(e)
        sliced = sliced[28:]
        print(errorentery.get())
        trail = len(errorentery.get()) + 18
        sliced = sliced[:-trail]
        text.set(sliced)

symbolButton = Button(window, text="Do Analysis", command=lambda: setsymbol(symbolentery.get() , errorentery.get()))
symbolButton.pack(pady=5)

window.protocol("WM_DELETE_WINDOW", on_closing)
window.mainloop()

