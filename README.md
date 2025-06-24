import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('/mnt/data/stroke.csv')

# Drop rows with missing values
df.dropna(inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Split features and labels
X = df.drop('stroke', axis=1)
y = df['stroke']

# 1. Split into train and test (3 marks)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Normalize data (5 marks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Feature Selection (5 marks)
# We'll use all features as-is for simplicity — alternatively, you can use SelectKBest or feature importance.

# 4. Fit the decision tree model (5 marks)
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# 5. Visualize the tree (5 marks)
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=["No Stroke", "Stroke"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()
