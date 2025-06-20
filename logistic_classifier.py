import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler
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

# Define input features for each variant
input_features_petal = [2, 3]
input_features_sepal = [0, 1]
input_features_all = [0, 1, 2, 3]

# Initialize classifiers for each variant
classifier1 = LogisticRegression()
classifier2 = LogisticRegression()
classifier3 = LogisticRegression()

# Train each classifier
classifier1.fit(X_train[:, input_features_petal], y_train, learning_rate=0.1)
classifier2.fit(X_train[:, input_features_sepal], y_train, learning_rate=0.1)
classifier3.fit(X_train[:, input_features_all], y_train, learning_rate=0.1)

classifier1.save("1_params_classifier.pkl")
classifier2.save("2_params_classifier.pkl")
classifier3.save("3_params_classifier.pkl")

# Visualize decision regions for petal length/width using mlxtend
plot_decision_regions(X_test[:, input_features_petal], y_test, clf=classifier1)
plt.title("Logistic Regression - Petal Length/Width")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

# Visualize decision regions for sepal length/width using mlxtend
plot_decision_regions(X_test[:, input_features_sepal], y_test, clf=classifier2)
plt.title("Logistic Regression - Sepal Length/Width")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()
