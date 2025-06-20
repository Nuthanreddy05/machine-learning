import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression

# Loading Iris dataset
iris_data = datasets.load_iris()
X = iris_data.data
y = iris_data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=37
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier3 = LogisticRegression()
classifier3.load("3_params_classifier.pkl")

# Evaluate accuracy on the test set
y_pred = classifier3.predict(X_test)
accuracy3 = np.mean(y_pred == y_test)


print("Accuracy - All:", accuracy3)
