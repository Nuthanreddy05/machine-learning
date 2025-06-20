import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Loading Iris dataset
iris_data = datasets.load_iris()
X = iris_data.data
y = iris_data.target

input_features_petal = [2, 3]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=37
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and load the best-trained model
classifier1 = LogisticRegression()
classifier1.load("1_params_classifier.pkl")

print(
    "Classifier 1 accuracy: ",
    classifier1.score(X_test[:, input_features_petal], y_test),
)
