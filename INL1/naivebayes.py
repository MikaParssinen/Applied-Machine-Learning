from sklearn.naive_bayes import GaussianNB

def run_naive_bayes(X_train, X_test, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return (y_pred, model)