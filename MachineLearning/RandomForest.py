import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from matplotlib.gridspec import GridSpec

# Generate synthetic data
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, 
                           n_redundant=0, n_clusters_per_class=1)

# Train Random Forest
forest = RandomForestClassifier(n_estimators=3, max_depth=2)
forest.fit(X, y)

# Set Seaborn style
sns.set(style="whitegrid")

# Create a figure with subplots
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 3, figure=fig)

# Subplot 1: Input data
ax0 = fig.add_subplot(gs[0, 0])
sns.scatterplot(ax=ax0, x=X[:, 0], y=X[:, 1], hue=y, palette='coolwarm', edgecolor='k', s=70)
ax0.set_title("Input Data")
ax0.set_xlabel("Feature 1")
ax0.set_ylabel("Feature 2")

# Subplot 2 and 3: Individual decision trees
for i, estimator in enumerate(forest.estimators_):
    ax = fig.add_subplot(gs[i//2, 1 + i%2])
    plot_tree(estimator, filled=True, feature_names=["Feature 1", "Feature 2"], 
              class_names=["Class 0", "Class 1"], ax=ax)
    ax.set_title(f"Decision Tree {i+1}")

# Subplot 4: Aggregated decision boundary
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = forest.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax4 = fig.add_subplot(gs[1, 0])
ax4.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
sns.scatterplot(ax=ax4, x=X[:, 0], y=X[:, 1], hue=y, palette='coolwarm', edgecolor='k', s=70)
ax4.set_title("Aggregated Decision Boundary")
ax4.set_xlabel("Feature 1")
ax4.set_ylabel("Feature 2")

plt.tight_layout()
plt.show()