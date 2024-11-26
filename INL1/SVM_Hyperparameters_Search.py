import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from sklearn.model_selection import GridSearchCV

data = pd.read_csv("./FIFA18_players_database/clean_data_normalized.csv")
X = data.drop(columns=['Preferred Positions'])
y = data['Preferred Positions']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_classifier = SVC(kernel='linear', class_weight='balanced')
svm_classifier.fit(X_train, y_train)
y_test_pred = svm_classifier.predict(X_test)
y_train_pred = svm_classifier.predict(X_train)

# Calculate test and train accuracy
acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_test_pred)

# Print accuracy with 3 decimals
print(f"Accuracy of train: {acc_train:.3f}")
print(f"Accuracy of test: {acc_test:.3f}")

param_grid = {
    'C': [0.1, 1, 2],           # Regularization parameter
    'kernel': ['linear', 'rbf'],      # Kernel type
    'gamma': ['scale', 'auto'],       # Kernel coefficient
    'class_weight': ['balanced', None]  # Class weights
}

grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)

# Perform grid search on training data
grid_search.fit(X_train, y_train)

# Best parameters and model
print("Best Hyperparameters:", grid_search.best_params_)
