def accuracy_scorer(y_test, y_pred):
    length = len(y_test)
    count = 0
    for i in range(length):
        if y_pred[i] in y_test[i]:
            count += 1
    return count/length