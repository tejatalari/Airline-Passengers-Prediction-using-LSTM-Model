
#  Airline Passengers Prediction using LSTM Model 

This repository contains a Long Short-Term Memory (LSTM) model implemented using TensorFlow and Keras to predict the number of airline passengers based on historical data. The project demonstrates the application of LSTM networks for time series forecasting, with a focus on scaling, model training, and evaluation.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

The project involves building an LSTM model to predict future airline passenger numbers using past data. The dataset is preprocessed, scaled, and then used to train the LSTM model. The predictions are evaluated and visualized to demonstrate the model's performance.

## Dataset

The dataset used in this project is the "International Airline Passengers" dataset, which consists of monthly totals of international airline passengers from 1949 to 1960. The dataset is a classic time series dataset that is widely used for time series forecasting problems.

## Installation

To run this project, you'll need to have Python installed along with the necessary libraries. You can install the required dependencies using pip:

```bash
git clone https://github.com/your-username/airline-passengers-lstm.git
cd airline-passengers-lstm
pip install numpy pandas matplotlib tensorflow scikit-learn
```

## Data Preparation

The dataset is loaded from a CSV file and preprocessed for training:

1. **Loading Data**: The data is loaded using `pandas` and converted into a NumPy array.
2. **Scaling**: The data is normalized using `MinMaxScaler` from `scikit-learn` to scale the values between 0 and 1.
3. **Splitting Data**: The data is split into training and testing sets.
4. **Reshaping Data**: The data is reshaped to fit the input format required by the LSTM model.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('airline-passengers.csv', usecols=[1])
df = df.values.astype('float32')
scaler = MinMaxScaler()
df = scaler.fit_transform(df)

train_size = int(len(df) * 0.67)
test_size = len(df) - train_size
train, test = df[0:train_size, :], df[train_size:len(df), :]

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
```

## Model Architecture

The LSTM model is defined using TensorFlow's Keras API. It consists of one LSTM layer followed by a Dense layer:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
```

## Training

The model is trained using the training dataset for 100 epochs with a batch size of 1. The training process minimizes the mean squared error loss:

```python
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
```

## Evaluation

The model's performance is evaluated using the Root Mean Squared Error (RMSE) on both the training and testing datasets. The predictions are also inverted back to the original scale:

```python
from sklearn.metrics import mean_squared_error
import math

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

print(f'Train Score: {trainScore:.2f} RMSE')
print(f'Test Score: {testScore:.2f} RMSE')
```

## Visualization

The predictions are plotted alongside the original data to visualize the model's performance:

```python
import matplotlib.pyplot as plt

trainPredictPlot = np.empty_like(df)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

testPredictPlot = np.empty_like(df)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(df) - 1, :] = testPredict

plt.plot(scaler.inverse_transform(df))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
```

## Usage

To run the project, simply execute the Python script after placing the dataset in the same directory. The script will train the model, make predictions, and plot the results.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

