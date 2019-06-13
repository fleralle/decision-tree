"""DecisionTree Class."""
import pandas as pd
from math import log2


class DecisionTree(object):
    """DecisionTree Class implements ID3 and CART."""

    def __init__(self, criterion='entropy'):
        """Decision Tree constructor.

        Parameters
        ----------
        criterion : str
            The criterion is set by default to 'entropy'. 2 possible values are
            'entropy' and 'gini'. Thee criterion is used when we calculate the
            Gain on each feature selection.

        """
        self.tree = None  # default model tree
        self.features = None  # default features array
        self.target = None  # default model target
        self.criterion = criterion

    def fit(self, X: pd.DataFrame, y: pd.Series, features: []):
        """Short summary.

        Parameters
        ----------
        X : pd.DataFrame
            Predictors pandas DataFrame.
        y : pd.Series
            Target true values DataSerie.
        features : []
            Array of `features` to use in model.

        Returns
        -------
        dict
            Dictionary object containing the tree representation.

        """
        data = pd.concat([X, y], axis=1)
        self.features = features
        self.target = y.name
        self.tree = self.build_tree(data, features)

        return self.tree

    def predict(self, X: pd.DataFrame):
        """Predict the target based on X features.

        Parameters
        ----------
        X : pd.DataFrame
            Predictors pandas DataFrame.

        Returns
        -------
        pd.Serie
            DataSerie corresponding to the actual predictions.

        """
        results = []
        for index, datapoint in X.iterrows():
            results.append(self.classify(self.tree, datapoint))

        return pd.Series(results, index=X.index, dtype='int', name=self.target)

    def accuracy(self, y_true: pd.Series, y_predicted: pd.Series):
        """Calculate the accuracy of the decision tree.

        Parameters
        ----------
        y_true : pd.Series
            The target true values.
        y_predicted : pd.Series
            The target predicted values.

        Returns
        -------
        float
            The actual DT accuracy rounded to 4 decimals.

        """
        y_check = y_true == y_predicted

        return round(y_check.sum() / len(y_check), 4)

    def classify(self, tree, datapoint):
        """Classify a specific data entry.

        Parameters
        ----------
        tree : {}
            Dictionary object containing the tree representation.
        datapoint : object
            datapoint containing the features.

        Returns
        -------
        type
            Corresponding class predicted associated to datapoint.

        """
        if type(tree) == dict:
            first_feature = list(tree.keys())[0]

            try:
                subtree = tree[first_feature][datapoint[first_feature]]
                return self.classify(subtree, datapoint)
            except Exception:
                return False
        else:
            return tree

    def build_tree(self, data: pd.DataFrame, features: []):
        """Build the DecisionTree representation.

        Parameters
        ----------
        data : pd.DataFrame
            Data used to build Decision Tree model.
        features : []
            Array of `features` to use in model.

        Returns
        -------
        {}
            Dictionary object containing the tree representation.

        """
        gains = []
        tree = {}

        # calculate default classe based on the most present class
        default = data[self.target].value_counts(sort=True).index[0]

        # Doing this as to prevent features alteration.
        copy_features = features.copy()

        if copy_features:
            for feature in copy_features:
                gains.append(self.calc_info_gain(data, feature))
            ft_gains = list(zip(features, gains))

            max = sorted(ft_gains, key=lambda x: x[1], reverse=True)[0]
            best_feature = max[0]
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

    def calc_entropy(self, p):
        """Calculate the entropy for a specific feature.

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

    def calc_info_gain(self, data: pd.DataFrame, feature: str):
        """Calculate the entropy gain in ID3 for a specific feature.

        Parameters
        ----------
        data : pd.DataFrame
            Data used to calculate the entropy gain.
        feature : str
            The feature name.

        Returns
        -------
        float
            The corresping entropy gain for the specified feature.

        """
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
        feat_entropy_sum = 0
        for feature_val, feature_stats in feature_values.items():
            feat_entropy = 0
            for feat_class, feat_count in feature_stats['classes'].items():
                feature_prob = feat_count / feature_stats['count']
                feat_entropy += self.calc_entropy(feature_prob)
            feat_entropy_sum += (feature_stats['count']/data_len)*feat_entropy

        # Calculate Entropy
        entropy = 0
        for class_name, class_count in classes.items():
            class_prob = class_count / data_len
            entropy += self.calc_entropy(class_prob)

        # Calc gain based on ID3 formule
        gain = entropy - feat_entropy_sum

        return gain

    def print_tree(self, tree: {}, prefix=''):
        """Print tree object representation.

        Parameters
        ----------
        tree : {}
            The tree dict to print.
        prefix : str
            Line prefix.

        Returns
        -------
        type
            Description of returned object.

        """
        if type(tree) == dict:
            for key in tree.keys():
                print(prefix, key)
                for item in tree[key].keys():
                    print(prefix, item)
                    self.print_tree(tree[key][item], prefix + "\t")
        else:
            print(prefix, "\t->\t", tree)

    def __print__(self):
        """Print tree representation."""
        self.print_tree(self.tree)
