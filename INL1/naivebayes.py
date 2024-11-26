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

def grid_search_naive_bayes(X_train, X_test, y_train, y_test, naive_bayes_model):
    params = [100.0, 50.0, 25.0, 10.0, 5.0, 2.5, 1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.0001]
    max_acc = 0
    best_param = 1

    for param in params:
        if naive_bayes_model == 'complement':
            model = ComplementNB(alpha=param*100_000, force_alpha=True)
        elif naive_bayes_model == 'multinomial':
            model = MultinomialNB(alpha=param*100_000, force_alpha=True)
        elif naive_bayes_model == 'bernoulli':
            model = BernoulliNB(alpha=param*100_000, force_alpha=True)
        else:
            print("Model does not support alpha parameter")
            return

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_scorer(y_test, y_pred)
        print("Acc = ", acc, " alpha = ", param * 100_000)

        if acc > max_acc:
            max_acc = acc
            best_param = param * 100_000

        del model
        del y_pred
        del acc
    return best_param