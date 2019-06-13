"""DecisionTree Class."""
import pandas as pd
from math import log2


class DecisionTree(object):
    """DecisionTree Class implements ID3 and CART."""

    def __init__(self):
        """Constructor"""
        self.tree = None
        self.features = None
        self.target = None

    def fit(self, X, y, features):
        data = pd.concat([X, y], axis=1)
        self.features = features
        self.target = y.name
        self.tree = self.build_tree(data, features)

        return self.tree

    def predict(self, X):
        results = []
        for index, datapoint in X.iterrows():
            results.append(self.classify(self.tree, datapoint))

        return pd.Series(results, index=X.index, dtype='int', name=self.target)

    def accuracy(self, y_true, y_predicted):
        y_check = y_true == y_predicted

        return round(y_check.sum() / len(y_check), 4)

    def classify(self, tree, datapoint):
        if type(tree) == dict:
            first_feature = list(tree.keys())[0]

            try:
                subtree = tree[first_feature][datapoint[first_feature]]
                return self.classify(subtree, datapoint)
            except Exception:
                return False
        else:
            return tree

    def build_tree(self, data, features):
        gains = []
        tree = {}

        # calculate default classe based on the most present class
        default = data[self.target].value_counts(normalize=True, sort=True).index[0]

        copy_features = features.copy()
        # print(features)
        if copy_features:
            for feature in copy_features:
                gains.append(self.calc_info_gain(data, feature))
            ft_gains = list(zip(features, gains))

            max = sorted(ft_gains, key=lambda x: x[1], reverse=True)[0]
            best_feature = max[0]
            # print(best_feature)
            best_feature_values = data[best_feature].unique()

            # Remove best feature from features list
            copy_features.remove(max[0])

            for best_feature_value in best_feature_values:
                feature_data = data[data[best_feature] == best_feature_value]
                # print(best_feature_value)
                subtree = self.build_tree(feature_data, copy_features)
                if best_feature in tree.keys():
                    tree[best_feature][best_feature_value] = subtree
                else:
                    tree[best_feature] = {}
                    tree[best_feature][best_feature_value] = subtree
            return tree
        else:
            return default

    def __print__(self):
        self.print_tree(self.tree)

    def print_tree(self, tree, str=''):
        if type(tree) == dict:
            for key in tree.keys():
                print(str, key)
                for item in tree[key].keys():
                    print(str, item)
                    self.print_tree(tree[key][item], str + "\t")
        else:
            print(str, "\t->\t", tree)

    def calc_entropy(self, p):
        """Short summary.

        Parameters
        ----------
        p : type
            Description of parameter `p`.

        Returns
        -------
        type
            Description of returned object.

        """
        if p != 0:
            return -p * log2(p)
        else:
            return 0

    def calc_info_gain(self, data, feature):
        gain = 0
        data_len = len(data)
        feature_values = {}
        classes = {}

        # Loop and get distinct feature value.
        for index, datapoint in data.iterrows():
            data_cls = datapoint[self.target]
            data_ft = datapoint[feature]
            if data_ft in feature_values.keys():
                feature_values[data_ft]['count'] += 1
                if data_cls in feature_values[data_ft]['classes'].keys():
                    feature_values[data_ft]['classes'][data_cls] += 1
                else:
                    feature_values[data_ft]['classes'][data_cls] = 1
            else:
                feature_values[data_ft] = {}
                feature_values[data_ft]['count'] = 1
                feature_values[data_ft]['classes'] = {}
                feature_values[data_ft]['classes'][data_cls] = 1

            # Count classes
            if data_cls in classes.keys():
                classes[data_cls] += 1
            else:
                classes[data_cls] = 1

        # Loop through all possible feature values and calculate corresponding
        # feature_value / class map.
        features_entropy_sum = 0
        for feature_val, feature_stats in feature_values.items():
            feature_entropy_sum = 0
            for feature_class, feature_class_count in feature_stats['classes'].items():
                feature_prob = feature_class_count / feature_stats['count']
                feature_entropy_sum += self.calc_entropy(feature_prob)
            features_entropy_sum += (feature_stats['count'] / data_len) * feature_entropy_sum

        # Calculate Entropy
        entropy = 0
        for class_name, class_count in classes.items():
            class_prob = class_count / data_len
            entropy += self.calc_entropy(class_prob)

        # Calc gain based on ID3 formule
        gain = entropy - features_entropy_sum

        return gain
