
import json
import xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from helper_func import *
import os


class model:
    def __init__(self):
        '''
        Init the model
        '''
        self.model  = xgboost.XGBRegressor(seed=0, subsample=0.8, colsample_bytree=0.8, learning_rate= 0.1, n_estimators= 150, max_depth=6, objective ='binary:logistic',eval_metric =roc_auc_score )
        self.hours_duration = 3
        self.best_features = []


    def load(self, dir_path):
        '''
        Edit this function to fit your model.

        This function should load the model that you trained on the train set.
        :param dir_path: A path for the folder the model is submitted
        '''
        model_name = 'XGB_model.json'
        model_file = os.path.join(dir_path, model_name)
        self.model.load_model(model_file)

        best_features_name = 'best_features.json'
        best_features_file = os.path.join(dir_path, best_features_name)
        with open(best_features_file, "r") as fp:
            self.best_features = json.load(fp)

    def predict(self, X):
        '''
        Edit this function to fit your model.

        This function should provide predictions of labels on (test) data.
        Make sure that the predicted values are in the correct format for the scoring
        metric.
        domain_name_feat_X, cls_name_feat_X, ts_feat_X : our code for add features to the data before prediction.
        :param X: is DataFrame with the columns - 'Datetime', 'URL', 'Domain_Name','Domain_cls1', 'Domain_cls2', 'Domain_cls3', 'Domain_cls4'.
        :return: a float value of the prediction for class 1.
        '''

        domain_name_feat_X = relative_domain(X[['Domain_Name']])
        domain_name_feat_X = rename_and_16_convert(domain_name_feat_X,'Domain')

        cls_name_feat_X = cls_proportion(X[['Domain_cls1', 'Domain_cls2', 'Domain_cls3', 'Domain_cls4']])
        cls_name_feat_X = rename_and_16_convert(cls_name_feat_X,'cls')

        ts_feat_X = avg_relative_entrances_device_id(X[['Datetime']], self.hours_duration)
        ts_feat_X = rename_and_16_convert(ts_feat_X,'ts')

        df_X = pd.concat([domain_name_feat_X, cls_name_feat_X, ts_feat_X], axis=1)

        df_X = corresponding_columns_training_set(self.best_features, df_X)

        y = self.model.predict(df_X[self.best_features])

        return y[0]
