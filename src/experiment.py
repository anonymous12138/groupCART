from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from Measure import *
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from GroupCART import *
from GroupCART_multiple_PA import *
from data_utils import *

LESS_IS_MORE = [False,False,False,False]

def run_exp(df, keyword, n_candidates=10, seed=2333):
    train, test = train_test_split(df, test_size=0.3, random_state=seed)
    valid, test = train_test_split(test, test_size=0.67, random_state=seed)

    scaler = MinMaxScaler()
    train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
    test = pd.DataFrame(scaler.transform(test), columns=test.columns)
    valid = pd.DataFrame(scaler.transform(valid), columns=valid.columns)

    X_train = train.loc[:, train.columns != 'Probability']
    y_train = train.loc[:, 'Probability']
    X_valid = valid.loc[:, valid.columns != 'Probability']
    y_valid = valid.loc[:, 'Probability']
    X_test = test.loc[:, test.columns != 'Probability']
    y_test = test.loc[:, 'Probability']
    sens_idx = list(X_train.columns).index(keyword)
    features = list(train.columns[:-1])

    prediction_candidates = []
    objective_scores = []

    for i in tqdm(range(n_candidates)):
        obj = []
        #         print("Tree Building ...")
        tree = Vfdt(features, delta=0.01, nmin=1, tau=0.5, weight=(i) / n_candidates, sens_idx=sens_idx)
        tree.update(X_train.values, y_train.values)
        y_pred_valid = tree.predict(X_valid.values)
        y_pred_test = tree.predict(X_test.values)
        prediction_candidates.append(y_pred_test)
        cm1 = confusion_matrix(y_valid, y_pred_valid)
        #         print(cm1)

        obj.append(np.round(accuracy_score(y_valid, y_pred_valid), 3))
        #         objective_scores.append(np.round(precision_score(y_valid, y_pred_valid),3))
        #         objective_scores.append(np.round(recall_score(y_valid, y_pred_valid),3))
        obj.append(np.round(f1_score(y_valid, y_pred_valid), 3))
        obj.append(1 - np.absolute(measure_final_score(valid, y_pred_valid, cm1, keyword, 'aod')))
        #         obj.append(1-np.absolute(measure_final_score(valid, y_pred_valid, cm1, keyword, 'eod')))
        #         obj.append(1-np.absolute(measure_final_score(valid, y_pred_valid, cm1, keyword, 'SPD')))
        obj.append(1 - np.absolute(measure_final_score(valid, y_pred_valid, cm1, keyword, 'DI')))
        #         obj.append(1-np.absolute(calculate_flip(tree,X_valid,keyword)))
        objective_scores.append(obj)
    winner_idx, distances = continuous_dom(objective_scores, LESS_IS_MORE, tol=0.01)
    print("Winners:", winner_idx)
    predictions = [prediction_candidates[j] for j in winner_idx]
    # ensemble_weights = np.array([1 - distances[j] for j in winner_idx])
    # ensemble_weights = ensemble_weights / ensemble_weights.sum()
    final_predictions = ensemble_predict(np.transpose(predictions))

    cm = confusion_matrix(y_test, final_predictions)
    print("======Final Metrics:")
    print("Acc: ", np.round(accuracy_score(y_test, final_predictions), 3))
    print("Precision: ", np.round(precision_score(y_test, final_predictions), 3))
    print("Recall: ", np.round(recall_score(y_test, final_predictions), 3))
    print("F1: ", np.round(f1_score(y_test, final_predictions), 3))
    print("AOD: ", np.absolute(measure_final_score(valid, y_pred_valid, cm1, keyword, 'aod')))
    print("EOD: ", np.absolute(measure_final_score(valid, y_pred_valid, cm1, keyword, 'eod')))
    print("SPD: ", np.absolute(measure_final_score(valid, y_pred_valid, cm1, keyword, 'SPD')))
    print("DI: ", np.absolute(measure_final_score(valid, y_pred_valid, cm1, keyword, 'DI')))

    metrics = []
    metrics.append(np.round(accuracy_score(y_test, final_predictions), 3))
    metrics.append(np.round(precision_score(y_test, final_predictions), 3))
    metrics.append(np.round(recall_score(y_test, final_predictions), 3))
    metrics.append(np.round(f1_score(y_test, final_predictions), 3))
    metrics.append(np.absolute(measure_final_score(test, final_predictions, cm, keyword, 'aod')))
    metrics.append(np.absolute(measure_final_score(test, final_predictions, cm, keyword, 'eod')))
    metrics.append(np.absolute(measure_final_score(test, final_predictions, cm, keyword, 'SPD')))
    metrics.append(np.absolute(measure_final_score(test, final_predictions, cm, keyword, 'DI')))
    return metrics
