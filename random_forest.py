import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # Check for stopping criteria
        if num_samples <= 1 or num_labels == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return np.full(num_samples, np.argmax(np.bincount(y)))

        # Find the best split
        best_split = self._best_split(X, y)
        if best_split is None:
            return np.full(num_samples, np.argmax(np.bincount(y)))

        feature_index, feature_value = best_split

        # Split the data
        left_indices = X[:, feature_index] < feature_value
        right_indices = ~left_indices

        # Build the left and right subtrees
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        # Combine the subtrees
        tree = np.zeros(num_samples, dtype=np.int64)
        tree[left_indices] = left_tree
        tree[right_indices] = right_tree

        return tree

    def _best_split(self, X, y):
        num_samples, num_features = X.shape
        best_score = np.inf
        best_split = None

        for feature_index in range(num_features):
            feature_values = np.unique(X[:, feature_index])

            for feature_value in feature_values:
                left_indices = X[:, feature_index] < feature_value
                right_indices = ~left_indices

                if np.any(left_indices) and np.any(right_indices):
                    y_left = y[left_indices]
                    y_right = y[right_indices]

                    score = (len(y_left) * self._gini(y_left) +
                             len(y_right) * self._gini(y_right))

                    if score < best_score:
                        best_score = score
                        best_split = (feature_index, feature_value)

        return best_split

    def _gini(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def predict(self, X):
        predictions = np.zeros(len(X), dtype=np.int64)
        for i, x in enumerate(X):
            node = self.tree
            while np.isscalar(node) == False:
                feature_index, feature_value = self._get_split(node)
                if x[feature_index] < feature_value:
                    node = node[:len(node) // 2]
                else:
                    node = node[len(node) // 2:]
            predictions[i] = node
        return predictions

    def _get_split(self, node):
        if np.isscalar(node):
            return None, None
        num_samples = len(self.tree)
        feature_index_array = np.where(node != 0)[0]
        if len(feature_index_array) == 0:
            return None, None
        feature_index = feature_index_array[0]
        feature_values = np.unique(self.tree)
        feature_value = feature_values[1]
        return feature_index, feature_value

# Random Forest Classifier
class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.estimators = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap, y_bootstrap = X[indices], y[indices]
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_bootstrap, y_bootstrap)
            self.estimators.append(tree)

    def predict(self, X):
        predictions = np.zeros((len(X), len(np.unique(y))))
        for tree in self.estimators:
            predictions[:, tree.predict(X)] += 1
        return np.argmax(predictions, axis=1)

# Train and evaluate the Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=3)
rf.fit(X_train, y_train)
accuracy = np.mean(rf.predict(X_test) == y_test)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")

# Visualize the decision boundaries
plt.figure(figsize=(10, 6))
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.8, edgecolor='k')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Random Forest Decision Boundary')
plt.show()