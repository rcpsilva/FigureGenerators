import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.svm import SVC

# Set Seaborn style
sns.set(style="whitegrid")

# Generate synthetic data
X, y = datasets.make_blobs(n_samples=100, centers=2, cluster_std=1.2)

# Fit the SVM
clf = SVC(kernel='linear', C=1.0)
clf.fit(X, y)

# Plot the decision boundary
plt.figure(figsize=(10, 6))

# Scatter plot using Seaborn
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='coolwarm', edgecolor='k', s=70)

# Get the separating hyperplane
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.7,
           linestyles=['--', '-', '--'])

# Plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=120, linewidth=1.5, facecolors='none', edgecolors='black', label='Support Vectors')

plt.title("SVM Decision Boundary with Margins and Support Vectors", fontsize=14)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.show()