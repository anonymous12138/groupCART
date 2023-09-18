import pandas as pd
import numpy as np


def flip(X_test,keyword):
    X_flip = X_test.copy()
    X_flip[keyword] = np.where(X_flip[keyword]==1, 0, 1)
    return X_flip


def calculate_flip(clf,X_test,keyword):
    X_flip = flip(X_test,keyword)
#     print(X_flip.columns)
    a = np.array(clf.predict(X_test))
    b = np.array(clf.predict(X_flip))
    total = X_test.shape[0]
    same = np.count_nonzero(a==b)
    return (total-same)/total


def continuous_dom(objective_scores, less_is_more, tol=0.01):
    PF_index = []
    n_goals = len(less_is_more)
    distances = []
    for i in range(len(objective_scores)):
        d2h = 0
        for j in range(n_goals):
            if less_is_more[j]:
                d2h += (objective_scores[i][j]) ** 2
            else:
                d2h += (1 - objective_scores[i][j]) ** 2
        distances.append((d2h) ** (1 / 2))
    temp = min(distances)
    for i in range(len(distances)):
        if abs(distances[i] - temp) <= tol:
            PF_index.append(i)
    return PF_index, distances


def _binary_dom(one, two):  # check, by d2h, if one is dominated by two
    not_equal = False
    for o, t in zip(one, two):
        if o > t:
            not_equal = True  # yes
        elif t > o:
            return False  # no
    return not_equal


def binary_dom(objective_scores):
    PF_index = []
    n_goals = len(objective_scores[0])
    dominated_counts = [0 for _ in range(len(objective_scores))]
    for i in range(len(objective_scores)):
        for j in range(len(objective_scores)):
            if i != j:
                if _binary_dom(objective_scores[i], objective_scores[j]):  # i dominated by j
                    dominated_counts[i] += 1
    for k in range(len(dominated_counts)):
        if dominated_counts[k] == 0:
            PF_index.append(k)
    return PF_index


def ensemble_predict(predictions,weights=None):
    final_prediction = []
    if weights is None:
        for i in range(len(predictions)):
            if np.mean(predictions[i])>=0.5:
                final_prediction.append(1)
            else:
                final_prediction.append(0)
    else:
        res = np.dot(predictions,weights)
        for i in range(len(res)):
            if res[i]>=0.5:
                final_prediction.append(1)
            else:
                final_prediction.append(0)
    return final_prediction