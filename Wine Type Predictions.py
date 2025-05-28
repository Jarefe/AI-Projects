import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import alpha
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')

red['type'] = 1
white['type'] = 0

wines = pd.concat([red, white], ignore_index=True) # Data concatenated into single dataframe
wines.dropna(inplace=True) # rows with missing data are dropped

# Graphing data

fig, ax = plt.subplots(1,2,figsize=(10,5))
ax[0].hist(wines[wines['type'] == 1].alcohol, bins=10,facecolor='red', edgecolor='black', lw=.5, alpha=.5, label='Red wine')
ax[1].hist(wines[wines['type']==0].alcohol, bins=10, facecolor='white', edgecolor='black', lw=.5, alpha=.5, label='White wine')

for a in ax:
    a.set_ylim([0,1000])
    a.set_xlabel('Alcohol in % Vol')
    a.set_ylabel('Frequency')

ax[0].set_title('Alcohol Content in Red Wine')
ax[1].set_title('Alcohol Content in White Wine')

fig.suptitle('Distribution of Alcohol by Wine Type')
plt.tight_layout()
plt.show()

# Splitting training and testing data

X = wines.iloc[:,:-1]
y = wines['type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.34, random_state=45)

# Creating neural network model
model = Sequential()
# ReLU allows positive values to pass through unchanged while setting all negative values to zero.
# https://www.geeksforgeeks.org/relu-activation-function-in-deep-learning/
model.add(Dense(12, activation='relu', input_dim=12)) # input layer
model.add(Dense(9, activation='relu')) # hidden layer

# Sigmoid maps values [0,1]
# https://www.geeksforgeeks.org/derivative-of-the-sigmoid-function/
model.add(Dense(1, activation='sigmoid')) # output layer

# Binary cross entropy quantifies the difference between actual class labels (0 or 1) and the predicted probabilities from the model
# https://www.geeksforgeeks.org/binary-cross-entropy-log-loss-for-binary-classification/
# Adam optimizer works well with large datasets and complex models
# https://www.geeksforgeeks.org/adam-optimizer/
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training model
model.fit(X_train, y_train, epochs=3, batch_size=1, verbose=1)


# Making predictions
y_pred = model.predict(X_test)

y_pred_labels = (y_pred >= .5).astype(int)

print(f"X test: {X_test}\n y_pred: {y_pred}")

for prediction in y_pred_labels[:12]:
    wine_type = "Red wine" if prediction == 1 else "White wine"
    print(f"Prediction: {wine_type}")