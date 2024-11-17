from sklearn.naive_bayes import GaussianNB

def run_gaussian_naive_bayes(X_train, X_test, y_train):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    return y_pred