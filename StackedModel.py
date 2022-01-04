import pandas as pd
import math
import os
import numerapi
import numpy as np
from IPython.display import clear_output

import keys
from splitting.utils import PurgedKFold

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


class NumerAIStack:
    def __init__(self, val_size,
                 model_type,
                 num_windows,
                 rf_n_estimators,
                 rf_max_features,
                 rf_min_samples_split,
                 gb_n_estimators,
                 gb_max_features,
                 l2_reg,
                 cv_split,
                 target_days,
                 embargo,
                 one_hot_encode
                 ):
        self.public_id = keys.public_id
        self.secret = keys.secret
        self.napi = numerapi.NumerAPI(self.public_id, self.secret)
        self.val_size = val_size
        self.model_type = model_type
        self.num_windows = num_windows + 1
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_features = rf_max_features
        self.rf_min_samples_split = rf_min_samples_split
        self.gb_n_estimators = gb_n_estimators
        self.gb_max_features = gb_max_features
        self.l2_reg = l2_reg
        self.cv_split = cv_split
        self.target_days = target_days
        self.embargo = embargo
        self.one_hot_encode = one_hot_encode
        self.load_data()

        self.eras = self.training_data.era.unique()
        self.splits = self.SplitList(self.eras, math.floor(len(self.eras) / self.num_windows))
        self.training_splits = self.splits[:-1]
        self.val_split = self.splits[-1]
        self.training_windows = [self.training_data[self.training_data['era'].isin(self.training_splits[i])] for i in
                                 range(len(self.training_splits))]
        self.last_validation_X = self.training_data[self.training_data['era'].isin(self.val_split)][
            [col for col in self.training_data.columns if 'feature' in col]]
        self.last_validation_y = self.cat_transform(
            self.training_data[self.training_data['era'].isin(self.val_split)][['target']])

    def load_data(self):
        if 'numerai_training_data.csv' not in os.listdir():
            napi = numerapi.NumerAPI(self.public_id, self.secret)

            # download data
            napi.download_current_dataset(unzip=True)
        else:
            print('Downloading Training Data')
            self.training_data = pd.read_csv('data/numerai_training_data.csv')
            self.training_data.era = [int(i.split('era')[1]) for i in self.training_data.era]
            print("Done")

        print('Downloading Tournament Data')
        # self.tournament_data= pd.read_parquet('numerai_tournament_data.parquet')
        print('Done')

    @staticmethod
    def cat_transform(target):
        category_map = {
            0.: 1,
            0.25: 2,
            0.5: 3,
            0.75: 4,
            1.: 5
        }
        return [category_map[tgt] for tgt in target['target']]

    @staticmethod
    def cat_back_transform(target):
        category_map = {
            1: 0.,
            2: 0.25,
            3: 0.5,
            4: 0.75,
            5: 1.0
        }
        return [category_map[tgt] for tgt in target]

    @staticmethod
    def SplitList(mylist, chunk_size):
        return [mylist[offs:offs + chunk_size] for offs in range(0, len(mylist), chunk_size)]

    def split_train_val(self, train, pct_train):

        if self.model_type == 'Regression':
            train_Y = train.iloc[:math.floor(len(train) * pct_train), :][['target']]
            val_Y = train.iloc[math.floor(len(train) * pct_train):, :][['target']]
        elif self.model_type == 'Classification':
            train_Y = self.cat_transform(train.iloc[:math.floor(len(train) * pct_train), :][['target']])
            val_Y = self.cat_transform(train.iloc[math.floor(len(train) * pct_train):, :][['target']])


        train_X = train.iloc[:math.floor(len(train) * pct_train), :][
            [col for col in train.columns if 'feature' in col]]
        val_X = train.iloc[math.floor(len(train) * pct_train):, :][
            [col for col in train.columns if 'feature' in col]]
        if self.one_hot_encode:
            encoder = OneHotEncoder(sparse = False)
            train_X = encoder.fit_transform(train_X)
            val_X = encoder.fit_transform(val_X)
        return train_X, train_Y, val_X, val_Y

    def build_random_forest_classifier(self):

        rfclass = RandomForestClassifier(n_estimators=self.rf_n_estimators, class_weight='balanced_subsample',
                                         criterion='entropy',
                                         min_samples_split = self.rf_min_samples_split)
        return rfclass

    def build_random_forest_regressor(self):

        rfreg = RandomForestRegressor(n_estimators=self.rf_n_estimators,
                                         min_samples_split = self.rf_min_samples_split)
        return rfreg

    def build_gb_classifier(self):

        clf = GradientBoostingClassifier(n_estimators=self.gb_n_estimators,
                                         max_features=self.gb_max_features,
                                         criterion='friedman_mse')

        return clf

    def build_gb_regressor(self):

        reg = GradientBoostingRegressor(n_estimators=self.gb_n_estimators,
                                         max_features=self.gb_max_features,
                                         criterion='friedman_mse')

        return reg

    def build_hist_classifier(self):
        clf = HistGradientBoostingClassifier(
            loss='categorical_crossentropy',
            max_iter=250,
            l2_regularization=self.l2_reg,
        )
        return clf

    def build_hist_regressor(self):
        reg = HistGradientBoostingRegressor(
            max_iter=250,
            l2_regularization=self.l2_reg,
        )
        return reg

    def build_sgd_classifier(self):
        clf = SGDClassifier(
            loss='log',
            class_weight='balanced',
            n_jobs=-1

        )
        return clf

    def build_sgd_regressor(self):
        reg = SGDRegressor(
            max_iter = 250,
            shuffle = False

        )
        return reg




    def build_stacking_classifier(self, rf, gbc, hcl, sgd):
        estimators_l1 = [
            ('gbc', gbc),
            ('hcl', hcl),
        ]

        estimators_l2 = [
            ('rf', rf),
            ('sgd', sgd)
        ]

        l2 = StackingClassifier(
            estimators=estimators_l2,
            final_estimator= ComplementNB(),
            n_jobs=-1,
        )

        clf = StackingClassifier(estimators=estimators_l1,
                                 final_estimator=l2,
                                 n_jobs = -1,
                                 passthrough = True
        )


        return clf

    def build_stacking_regressor(self, rf, gbc, hcl, sgd):
        estimators_l1 = [
            ('gbc', gbc),
            ('hcl', hcl)
        ]

        estimators_l2 = [
            ('rf', rf),
            ('sgd', sgd)
        ]

        l2 = StackingRegressor(
            estimators=estimators_l2,
            final_estimator= Ridge(max_iter = 250),
            n_jobs=-1,
        )

        reg = StackingRegressor(estimators=estimators_l1,
                                 final_estimator=l2,
                                 n_jobs = -1,
                                 passthrough = True
        )


        return reg

    def validate(self, model, X, Y):
        pred = model.predict(X)
        actual = Y

        cm = confusion_matrix(actual, pred)
        acc = accuracy_score(actual, pred)
        prec = precision_score(actual, pred, average='weighted')
        f1 = f1_score(actual, pred, average='weighted')

        return cm, acc, prec, f1

    def window_train(self, windows, pct_train):
        models = {}
        n = len(windows)
        stacked_model = None

        for idx, window in enumerate(windows):
            print(f'{idx / n * 100} % done training')
            clear_output(wait=True)
            train_X, train_Y, val_X, val_Y, = self.split_train_val(train=window,
                                                                  pct_train=pct_train)

            if self.model_type == 'Classification':
                rf_class = self.build_random_forest_classifier()
                gb_class = self.build_gb_classifier()
                hs_class = self.build_hist_classifier()
                sgd_class = self.build_sgd_classifier()
                stacked_model = self.build_stacking_classifier(rf_class, gb_class, hs_class, sgd_class)
                if self.one_hot_encode:
                    purged_k_fold = PurgedKFold(t1=pd.DataFrame(train_X).index.to_series(), n_splits=self.cv_split,
                                                pctEmbargo=self.embargo)
                    splits = purged_k_fold.split(pd.DataFrame(train_X).index.to_series(), pd.Series(train_Y).index.to_series())
                else:
                    purged_k_fold = PurgedKFold(t1=pd.Series(train_X.index), n_splits=self.cv_split, pctEmbargo = self.embargo)
                    splits = purged_k_fold.split(pd.Series(train_X.index), pd.Series(train_Y).index.to_series())

            elif self.model_type == 'Regression':
                rf_reg = self.build_random_forest_regressor()
                gb_reg = self.build_gb_regressor()
                hs_reg = self.build_hist_regressor()
                sgd_reg = self.build_sgd_regressor()
                stacked_model= self.build_stacking_regressor(rf_reg, gb_reg, hs_reg, sgd_reg)
                purged_k_fold = PurgedKFold(t1=pd.Series(train_X.index), n_splits=self.cv_split, pctEmbargo= self.embargo)
                splits = purged_k_fold.split(pd.Series(train_X.index), train_Y.index.to_series())
            stacked_model.cv = splits
            #stacking_class.cv = self.PurgedSplit.split(X=train_X)
            stacked_model.fit(train_X, train_Y)
            # model = self.train_random_forest(n_estimators, max_features, train_X, train_Y)
            cm, acc, precision, f1 = self.validate(stacked_model, val_X, val_Y)
            models[idx] = {'model': stacked_model,
                           'confusion': cm,
                           'accuracy': acc,
                           'precision': precision,
                           'f1': f1}
        return models

    def meta_predict(self, models, X):

        n = len(models.keys())
        preds = {}
        for idx_, key in enumerate(models.keys()):
            print(f'{idx_ / n * 100} % done predicting')
            clear_output(wait=True)
            model = models[key]['model']
            # if metaPred is None:
            keys = preds.keys()
            metaPred = model.predict(X)
            for idx, pred in enumerate(metaPred):
                if idx not in keys:
                    preds[idx] = [pred] * math.floor((models[key]['accuracy'] / sum_of_accuracy) * 1000)
                else:
                    preds[idx].extend([pred] * math.floor((models[key]['accuracy'] / sum_of_accuracy) * 1000))

        ensemblePrediction = [np.mean(self.cat_back_transform(preds[key])[0]) for key in preds.keys()]

        df = pd.DataFrame(ensemblePrediction, index=self.tournament_data.index, columns=['prediction'])
        df.to_csv('predictions.csv')
        return ensemblePrediction

    def tournament_predict(self, models):
        preds = self.meta_predict(models=models,
                                  X=self.tournament_data[[x for x in self.tournament_data.columns if 'feature' in x]])
        output = pd.DataFrame(preds).set_index(self.tournament_data.index)

        return output

    def submit_predictions(self, file, model_id):

        self.napi.upload_predictions(file, model_id=model_id)