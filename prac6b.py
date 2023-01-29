# Common imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Import the data set
raw_data = pd.read_csv(
    'Classified Data.csv',
    index_col=0)

# Standardize the data set
scaler = StandardScaler()
scaler.fit(raw_data.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(raw_data.drop('TARGET CLASS', axis=1))
scaled_data = pd.DataFrame(scaled_features, columns=raw_data.drop('TARGET CLASS', axis=1).columns)

# Split the data set into training data and test data

x = scaled_data
y = raw_data['TARGET CLASS']
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size=0.3)

# Train the model and make predictions
model = KNeighborsClassifier(n_neighbors=1)
model.fit(x_training_data, y_training_data)
predictions = model.predict(x_test_data)

# Performance measurement
print(classification_report(y_test_data, predictions))
print(confusion_matrix(y_test_data, predictions))

# Selecting an optimal K value
error_rates = []

# for i in np.arange(1, 101):
new_model = KNeighborsClassifier(n_neighbors=1)
new_model.fit(x_training_data, y_training_data)
new_predictions = new_model.predict(x_test_data)
error_rates.append(np.mean(new_predictions != y_test_data))
plt.figure(figsize=(16, 12))
plt.plot(error_rates)
plt.show()