import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Sample data
data = np.array([1.28, 2.76, 2.24, 1.15, 5.10, 11.02])
X = []
y = []
for i in range(len(data)-1):
    X.append(data[i:i+1])
    y.append(data[i+1])
X = np.array(X).reshape((len(X), 1, 1))

# Define LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Fit model
model.fit(X, y, epochs=200, verbose=0)

# Make prediction
x_input = np.array([[data[-1]]]).reshape((1, 1, 1))
predicted_value = model.predict(x_input)
print(predicted_value)
