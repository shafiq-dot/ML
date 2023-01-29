import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score

import warnings

warnings.filterwarnings('ignore')

dataset = pd.read_csv("diabetes.csv")

print(dataset.head())

x = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values

y = dataset.iloc[:, [-1]].values

print(x)

print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)

print(x_train[0:15, :])

classifier = LogisticRegression()

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix :\n ", cm)

print("Accuracy :", accuracy_score(y_test, y_pred))
