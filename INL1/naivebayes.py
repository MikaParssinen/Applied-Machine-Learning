from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

def accuracy_scorer(y_test, y_pred):
    length = len(y_test)
    count = 0
    for i in range(length):
        if y_pred[i] in y_test[i]:
            count += 1
    return count/length

def run_naive_bayes(X_train, X_test, y_train, naive_bayes_model):
    if naive_bayes_model == 'gaussian':
        model = GaussianNB()
    elif naive_bayes_model == 'complement':
        model = ComplementNB()
    elif naive_bayes_model == 'multinomial':
        model = MultinomialNB()
    else:
        model = BernoulliNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred