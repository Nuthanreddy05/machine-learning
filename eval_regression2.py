from sklearn import datasets
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.data[:, [0]]


# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=37
)


# Load model parameters
model = LinearRegression()
model.load("model2_regression.json")

# Test the model on the test set
X_test_comb_in = X_test
X_test_comb_out = y_test
mse = model.score(X_test_comb_in, X_test_comb_out)

# Print mean squared error
print(f"Mean Squared Error for Model 2: {mse}")
