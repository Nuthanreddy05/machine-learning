import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression

# Loading Iris dataset
iris_data = datasets.load_iris()
X = iris_data.data
y = iris_data.target

input_features_petal = [0, 1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=37
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and load the best-trained model
classifier2 = LogisticRegression()
classifier2.load("2_params_classifier.pkl")

y_pred = classifier2.predict(X_test[:, input_features_petal])
accuracy2 = np.mean(y_pred == y_test)

print("Accuracy - Sepal Length/Width:", accuracy2)
