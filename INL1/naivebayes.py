from sklearn.naive_bayes import GaussianNB

def accuracy_scorer(y_test, y_pred):
    length = len(y_test)
    count = 0
    for i in range(length):
        if y_pred[i] in y_test[i]:
            count += 1
    return count/length

def run_naive_bayes(X_train, X_test, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return (y_pred, model)