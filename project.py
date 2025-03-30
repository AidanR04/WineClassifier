import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_val_score

#Loading the dataset
df = pd.read_csv("winequality-red.csv")
df['quality'] = df['quality'].replace({3:0, 4:0, 5:0, 6:1, 7:1, 8:1})
print(df['quality'].value_counts())
#Separating features and target class
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

#First model will be a dihcision tree
clf = DecisionTreeClassifier(random_state=42)
#Training the classifier
clf.fit(X, Y)

accuracy_score = (cross_val_score(clf, X, Y, cv=10, scoring='accuracy')).mean()
precision_score = (cross_val_score(clf, X, Y, cv=10, scoring='precision')).mean()
recall_score = (cross_val_score(clf, X, Y, cv=10, scoring='recall')).mean()
f1_score = (cross_val_score(clf, X, Y, cv=2, scoring='f1')).mean()


print(f'Average Accuracy Score: {accuracy_score}')
print(f'Average Precision Score: {precision_score}')
print(f'Average Recall Score: {recall_score}')
print(f'Average F1 Score: {f1_score}')

plt.figure(figsize=(12, 8))  # Adjust the size as needed
plot_tree(clf, 
          filled=True,        # Color the nodes
          feature_names=X.columns,  # Use your feature names
          class_names=['Low Quality Wine', 'High Quality Wine'],  # Class names if binary classification
          rounded=True,       # Round the corners of the nodes
          fontsize=10)        # Font size for labels
plt.show()
