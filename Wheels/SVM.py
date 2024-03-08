from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Generate a random binary classification problem.
data, target = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=3)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=3)

# Initialize the SVM classifier with a linear kernel
svm_classifier = SVC(kernel='linear')

# Fit the classifier on the training data
svm_classifier.fit(X_train, y_train)

# Predict on the test data
predictions = svm_classifier.predict(X_test)

# Print out the classification report and accuracy score
print(classification_report(y_test, predictions))
print('Accuracy:', accuracy_score(y_test, predictions))

# Plotting the decision boundary, the margins, and the support vectors
plt.figure(figsize=(10,6))

# Create grid to evaluate model
xx = np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 30)
yy = np.linspace(X_train[:, 1].min() - 1, X_train[:, 1].max() + 1, 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm_classifier.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margins
plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
             linestyles=['--', '-', '--'])

# Plot support vectors
plt.scatter(svm_classifier.support_vectors_[:, 0], svm_classifier.support_vectors_[:, 1], s=100,
             linewidth=1, facecolors='none', edgecolors='k')

# Plot data points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='autumn')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap='autumn', marker='*')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Classifier with Linear Kernel')
plt.show()