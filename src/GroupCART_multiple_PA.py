import numpy as np
import pandas as pd
import time
from itertools import combinations
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils import check_array, check_X_y

class Multi_VfdtNode:
    def __init__(self, possible_split_features, weight=[1, 0, 0], sens_idx=[0]):
        """
        nijk: statistics of feature i, value j, class
        possible_split_features: features list
        """
        self.parent = None
        self.left_child = None
        self.right_child = None
        self.split_feature = None
        self.split_value = None  # both continuous and discrete value
        self.new_examples_seen = 0
        self.total_examples_seen = 0
        self.class_frequency = {}
        self.sens_frequency1 = {}
        self.sens_frequency2 = {}
        self.sens_idx = sens_idx
        self.nijk = {f: {} for f in possible_split_features}
        self.nijk_sens1 = {f: {} for f in possible_split_features}
        self.nijk_sens2 = {f: {} for f in possible_split_features}
        self.possible_split_features = possible_split_features
        #         self.weight=weight
        w_sum = sum(weight)
        self.weight = [each / w_sum for each in weight]

    def add_children(self, split_feature, split_value, left, right):
        self.split_feature = split_feature
        self.split_value = split_value
        self.left_child = left
        self.right_child = right
        left.parent = self
        right.parent = self

        self.nijk.clear()  # reset stats
        if isinstance(split_value, list):
            left_value = split_value[0]
            right_value = split_value[1]
            # discrete split value list's length = 1, stop splitting
            if len(left_value) <= 1:
                new_features = [None if f == split_feature else f
                                for f in left.possible_split_features]
                left.possible_split_features = new_features
            if len(right_value) <= 1:
                new_features = [None if f == split_feature else f
                                for f in right.possible_split_features]
                right.possible_split_features = new_features

    def is_leaf(self):
        return self.left_child is None and self.right_child is None

    # recursively trace down the tree
    # to distribute data examples to corresponding leaves
    def sort_example(self, x):
        if self.is_leaf():
            return self
        else:
            index = self.possible_split_features.index(self.split_feature)
            value = x[index]
            split_value = self.split_value

            if isinstance(split_value, list):  # discrete value
                if value in split_value[0]:
                    return self.left_child.sort_example(x)
                else:
                    return self.right_child.sort_example(x)
            else:  # continuous value
                if value <= split_value:
                    return self.left_child.sort_example(x)
                else:
                    return self.right_child.sort_example(x)

    # the most frequent class
    def most_frequent(self):
        try:
            prediction = max(self.class_frequency,
                             key=self.class_frequency.get)
        except ValueError:
            # if self.class_frequency dict is empty, go back to parent
            class_frequency = self.parent.class_frequency
            prediction = max(class_frequency, key=class_frequency.get)
        return prediction

    # update leaf stats in order to calculate gini
    def update_stats(self, x, y):
        feats = self.possible_split_features
        #         nijk = self.nijk
        iterator = [f for f in feats if f is not None]
        for i in iterator:
            value = x[feats.index(i)]
            if value not in self.nijk[i]:
                self.nijk[i][value] = {y: 1}
            else:
                try:
                    self.nijk[i][value][y] += 1
                except KeyError:
                    self.nijk[i][value][y] = 1

            if value not in self.nijk_sens1[i]:
                self.nijk_sens1[i][value] = {x[self.sens_idx[0]]: 1}
            else:
                try:
                    self.nijk_sens1[i][value][x[self.sens_idx[0]]] += 1
                except KeyError:
                    self.nijk_sens1[i][value][x[self.sens_idx[0]]] = 1

            if value not in self.nijk_sens2[i]:
                self.nijk_sens2[i][value] = {x[self.sens_idx[1]]: 1}
            else:
                try:
                    self.nijk_sens2[i][value][x[self.sens_idx[1]]] += 1
                except KeyError:
                    self.nijk_sens2[i][value][x[self.sens_idx[1]]] = 1

        self.total_examples_seen += 1
        self.new_examples_seen += 1

        # update freq of sensitive attribute

        try:
            self.sens_frequency1[x[self.sens_idx[0]]] += 1
        except KeyError:
            self.sens_frequency1[x[self.sens_idx[0]]] = 1

        try:
            self.sens_frequency2[x[self.sens_idx[1]]] += 1
        except KeyError:
            self.sens_frequency2[x[self.sens_idx[1]]] = 1

        # update freq of target attribute
        try:
            self.class_frequency[y] += 1
        except KeyError:
            self.class_frequency[y] = 1

    def check_not_splitting_sens(self):
        X1 = 1
        sens_frequency1 = self.sens_frequency1
        n = sum(sens_frequency1.values())
        for j, k in sens_frequency1.items():
            X1 -= (k / n) ** 2

        X2 = 1
        sens_frequency2 = self.sens_frequency2
        n = sum(sens_frequency2.values())
        for j, k in sens_frequency2.items():
            X2 -= (k / n) ** 2

        return self.weight[1] * (.5 - X1) + self.weight[2] * (.5 - X2)

    def check_not_splitting(self):
        # compute gini index for not splitting
        X0 = 1
        class_frequency = self.class_frequency
        n = sum(class_frequency.values())
        for j, k in class_frequency.items():
            X0 -= (k / n) ** 2
        return (self.weight[0]) * X0

    # use Hoeffding tree model to test node split, return the split feature
    def attempt_split(self, delta, nmin, tau):
        if self.new_examples_seen < nmin:
            return None
        class_frequency = self.class_frequency
        sens_frequency1 = self.sens_frequency1
        sens_frequency2 = self.sens_frequency2
        if len(class_frequency) == 1:
            return None
        self.new_examples_seen = 0  # reset
        nijk = self.nijk
        nijk_sens1 = self.nijk_sens1
        nijk_sens2 = self.nijk_sens2
        MIN = 1
        second_min = 1
        Xa = ''
        split_value = None
        for feature in self.possible_split_features:
            if feature is not None:
                njk = nijk[feature]
                njk_sens1 = nijk_sens1[feature]
                njk_sens2 = nijk_sens2[feature]
                gini, value = self.gini(njk, njk_sens1, njk_sens2, class_frequency, sens_frequency1, sens_frequency2)
                #                 gini_s, value_s = self.gini_sensitive(njk, sens_frequency)
                #                 gini_combined = (1-self.weight)*gini + self.weight*gini_s
                if gini < MIN:
                    MIN = gini
                    Xa = feature
                    split_value = value
                elif MIN < gini < second_min:
                    second_min = gini

        epsilon = self.hoeffding_bound(delta)
        g_X0 = self.check_not_splitting() + self.check_not_splitting_sens()
        if MIN < g_X0:
            # print(second_min - min, epsilon)
            if second_min - MIN > epsilon:
                # print('1 node split')
                return [Xa, split_value]
            elif tau != 0 and second_min - MIN < epsilon < tau:
                # print('2 node split')
                return [Xa, split_value]
            else:
                return None
        return None

    def hoeffding_bound(self, delta):
        n = self.total_examples_seen
        R = np.log(len(self.class_frequency))
        return np.sqrt(R * R * np.log(1 / delta) / (2 * n))

    def gini(self, njk, njk_sens1, njk_sens2, class_frequency, sens_frequency1=None, sens_frequency2=None):
        # gini(D) = 1 - Sum(pi^2)
        # gini(D, F=f) = |D1|/|D|*gini(D1) + |D2|/|D|*gini(D2)

        D = self.total_examples_seen
        m1 = 1  # minimum gini
        m2 = 1  # second minimum gini
        Xa_value = None
        feature_values = list(njk.keys())  # list() is essential
        if not isinstance(feature_values[0], str):  # numeric  feature values
            sort = np.array(sorted(feature_values))
            # vectorized computation, like in R
            split = (sort[0:-1] + sort[1:]) / 2

            D1_class_frequency = {j: 0 for j in class_frequency.keys()}
            D1_sens_frequency1 = {j: 0 for j in sens_frequency1.keys()}
            D1_sens_frequency2 = {j: 0 for j in sens_frequency2.keys()}
            for index in range(len(split)):
                nk = njk[sort[index]]
                nk_sens1 = njk_sens1[sort[index]]
                nk_sens2 = njk_sens2[sort[index]]
                for j in nk:
                    D1_class_frequency[j] += nk[j]

                for j in nk_sens1:
                    D1_sens_frequency1[j] += nk_sens1[j]

                for j in nk_sens2:
                    D1_sens_frequency2[j] += nk_sens2[j]

                D1 = sum(D1_class_frequency.values())
                D2 = D - D1
                g_d1 = 1
                g_d2 = 1

                D1_sens1 = sum(D1_sens_frequency1.values())
                D2_sens1 = D - D1_sens1

                D1_sens2 = sum(D1_sens_frequency2.values())
                D2_sens2 = D - D1_sens2

                g_d1_sens1 = 1
                g_d2_sens1 = 1
                g_d1_sens2 = 1
                g_d2_sens2 = 1

                D2_class_frequency = {}
                D2_sens_frequency1 = {}
                D2_sens_frequency2 = {}

                for key, value in class_frequency.items():
                    if key in D1_class_frequency:
                        D2_class_frequency[key] = value - D1_class_frequency[key]
                    else:
                        D2_class_frequency[key] = value

                for key, v in D1_class_frequency.items():
                    g_d1 -= (v / D1) ** 2
                for key, v in D2_class_frequency.items():
                    g_d2 -= (v / D2) ** 2
                g = g_d1 * D1 / D + g_d2 * D2 / D

                # Important adjust!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                for key, value in sens_frequency1.items():
                    if key in D1_sens_frequency1:
                        D2_sens_frequency1[key] = value - D1_sens_frequency1[key]
                    else:
                        D2_sens_frequency1[key] = value

                for key, v in D1_sens_frequency1.items():
                    g_d1_sens1 -= (v / D1_sens1) ** 2
                for key, v in D2_sens_frequency1.items():
                    g_d2_sens1 -= (v / D2_sens1) ** 2
                g_sens1 = g_d1_sens1 * D1_sens1 / D + g_d2_sens1 * D2_sens1 / D

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                for key, value in sens_frequency2.items():
                    if key in D1_sens_frequency2:
                        D2_sens_frequency2[key] = value - D1_sens_frequency2[key]
                    else:
                        D2_sens_frequency2[key] = value

                for key, v in D1_sens_frequency2.items():
                    g_d1_sens2 -= (v / D1_sens2) ** 2
                for key, v in D2_sens_frequency2.items():
                    g_d2_sens2 -= (v / D2_sens2) ** 2
                g_sens2 = g_d1_sens2 * D1_sens2 / D + g_d2_sens2 * D2_sens2 / D

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                g_combined = (self.weight[0]) * g + self.weight[1] * (.5 - g_sens1) + self.weight[2] * (.5 - g_sens2)
                if g_combined < m1:
                    m1 = g_combined
                    Xa_value = split[index]
                # elif m1 < g < m2:
                # m2 = g
            return [m1, Xa_value]

        else:  # discrete feature_values
            print("HIT!!!!")
            length = len(njk)
            if length > 10:  # too many discrete feature values, estimate
                for j, k in njk.items():
                    D1 = sum(k.values())
                    D2 = D - D1
                    g_d1 = 1
                    g_d2 = 1

                    D2_class_frequency = {}
                    for key, value in class_frequency.items():
                        if key in k:
                            D2_class_frequency[key] = value - k[key]
                        else:
                            D2_class_frequency[key] = value
                    for key, v in k.items():
                        g_d1 -= (v / D1) ** 2

                    if D2 != 0:
                        for key, v in D2_class_frequency.items():
                            g_d2 -= (v / D2) ** 2
                    g = g_d1 * D1 / D + g_d2 * D2 / D
                    if g < m1:
                        m1 = g
                        Xa_value = j
                    # elif m1 < g < m2:
                    # m2 = g
                right = list(np.setdiff1d(feature_values, Xa_value))

            else:  # fewer discrete feature values, get combinations
                print("HIT!!!!")
                comb = self.select_combinations(feature_values)
                for i in comb:
                    left = list(i)
                    D1_class_frequency = {
                        key: 0 for key in class_frequency.keys()}
                    D2_class_frequency = {
                        key: 0 for key in class_frequency.keys()}
                    for j, k in njk.items():
                        for key, value in class_frequency.items():
                            if j in left:
                                if key in k:
                                    D1_class_frequency[key] += k[key]
                            else:
                                if key in k:
                                    D2_class_frequency[key] += k[key]
                    g_d1 = 1
                    g_d2 = 1
                    D1 = sum(D1_class_frequency.values())
                    D2 = D - D1
                    for key, v in D1_class_frequency.items():
                        g_d1 -= (v / D1) ** 2
                    for key, v in D2_class_frequency.items():
                        g_d2 -= (v / D2) ** 2
                    g = g_d1 * D1 / D + g_d2 * D2 / D
                    if g < m1:
                        m1 = g
                        Xa_value = left
                    # elif m1 < g < m2:
                    # m2 = g
                right = list(np.setdiff1d(feature_values, Xa_value))
            return [m1, [Xa_value, right]]

    # divide values into two groups, return the combination of left groups
    def select_combinations(self, feature_values):
        combination = []
        e = len(feature_values)
        if e % 2 == 0:
            end = int(e / 2)
            for i in range(1, end + 1):
                if i == end:
                    cmb = list(combinations(feature_values, i))
                    enough = int(len(cmb) / 2)
                    combination.extend(cmb[:enough])
                else:
                    combination.extend(combinations(feature_values, i))
        else:
            end = int((e - 1) / 2)
            for i in range(1, end + 1):
                combination.extend(combinations(feature_values, i))

        return combination


# very fast decision tree class, i.e. hoeffding tree
class Multi_Vfdt:
    def __init__(self, features, delta=0.01, nmin=100, tau=0.1, weight=[3, 0, 0], sens_idx=[0, 0]):
        """
        :features: list of data features
        :delta: used to compute hoeffding bound, error rate
        :nmin: to limit the G computations
        :tau: to deal with ties
        """
        self.features = features
        self.delta = delta
        self.nmin = nmin
        self.tau = tau
        self.root = Multi_VfdtNode(features, weight=weight, sens_idx=sens_idx)
        self.n_examples_processed = 0
        w_sum = sum(weight)
        self.weight = [each / w_sum for each in weight]
        self.sens_idx = sens_idx

    # update the tree by adding one or many training example(s)
    def update(self, X, y):
        X, y = check_X_y(X, y)
        for x, _y in zip(X, y):
            self.__update(x, _y)
        # print("Update!")
        # print("X {}: {}".format(type(X), X))
        # print("y {}: {}".format(type(y), y))
        # self.print_tree()
        # print("---")
        # if isinstance(y, (np.ndarray, list)):
        #     for x, _y in zip(X, y):
        #         self.__update(x, _y)
        # else:
        #     self.__update(X, y)
        # self.print_tree()
        # print("---")
        # print("End update! n_examples_processed={}".format(
        #     self.n_examples_processed))
        # print("--- --- ---")

    # update the tree by adding one training example
    def __update(self, x, _y):
        self.n_examples_processed += 1
        node = self.root.sort_example(x)
        node.update_stats(x, _y)

        result = node.attempt_split(self.delta, self.nmin, self.tau)
        if result is not None:
            feature = result[0]
            value = result[1]
            self.node_split(node, feature, value)

    # split node, produce children
    def node_split(self, node, split_feature, split_value):
        features = node.possible_split_features
        # print('node_split')
        left = Multi_VfdtNode(features, weight=self.weight, sens_idx=self.sens_idx)
        right = Multi_VfdtNode(features, weight=self.weight, sens_idx=self.sens_idx)
        node.add_children(split_feature, split_value, left, right)

    # predict test example's classification
    def predict(self, X):
        X = check_array(X)
        return [self.__predict(x) for x in X]
        # if isinstance(X, (np.ndarray, list)):
        #     return [self.__predict(x) for x in X]
        # else:
        #     leaf = self.__predict(X)

    def __predict(self, x):
        leaf = self.root.sort_example(x)
        return leaf.most_frequent()

    def print_tree(self, node=None, level=1):
        if node is None:
            self.print_tree(self.root)
        elif node.is_leaf():
            level = level
        #             print('Leaf')
        else:
            print("-" * level, node.split_feature)
            self.print_tree(node.left_child, level + 1)
            self.print_tree(node.right_child, level + 1)


def calc_metrics(y_test, y_pred, row_name):
    accuracy = accuracy_score(y_test, y_pred)
    metrics = list(
        precision_recall_fscore_support(
            y_test, y_pred, average='weighted',
            labels=np.unique(y_pred)))
    metrics = pd.DataFrame({
        'accuracy': accuracy,
        'precision': metrics[0],
        'recall': metrics[1],
        'f1': metrics[2]}, index=[row_name])
    return metrics