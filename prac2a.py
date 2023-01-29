import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Reading dataset using pandas
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
dataset = pd.read_csv(url, names=names)

# Display dataset
print(dataset.head(10))
X = dataset.drop("class", 1)
Y = dataset["class"]

# Splitting the train and test datasets

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Display Training and testing data

print(f"\nDataSet before PCA :\n\nTrain :\n{x_train}\n\nTest :\n{x_test}")
# print(x_train)
# print(x_test)

# Creating PCA
pca = PCA()
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# Giving a principal feature to model
pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

print(f"\nDataSet After PCA :\n\nTrain :\n{x_train}\n\nTest :\n{x_test}")