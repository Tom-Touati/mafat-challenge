import xgboost
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json
from submission.prepare_and_train_model import *
from submission.utils import *
from submission.domain_timeseries_processing import *
from submission.content_based_features import *
from submission.frequency_base_feats import *
from submission.cls_features import *
from submission.load_and_prepare_input import *
# from domain_timeseries_processing import *
# from utils import *
# from load_and_prepare_input import *
# from prepare_and_train_model import *
# from content_based_features import *
# from frequency_base_feats import *
# from cls_features import *

PARAMS = {
    'seed': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'learning_rate': 0.1,
    'max_depth': 3,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'weighted_final_scores': {},
    'mean_probability_score': {
        'fillna': True,
        'square_usage': True,
        'norm': 'z'
    },
    'feature_selection': {
        'n_features_to_select': 2000,
        'step': 20000
    },
    'model': {
        'seed': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'learning_rate': 0.1,
        'n_estimators': 250,
        'max_depth': 4,
        'objective': 'binary:logistic'
    },
    'no-normalization-finalscores': True,
    "user_activity_timeseries": {
        "bin_hours": 6,
        "gaussian_filter": True,
        "n_days_each_side": 3,
        "std": 1.5,
        "drop_na": True,
        "drop_zeros": False
    }
}


class model:

    def __init__(self, params=PARAMS, engine=pd):
        '''
        Init the model
        '''
        self.model = xgboost.XGBRegressor()
        self.domain_activity = None
        self.user_activity = None
        self.best_features = None
        self.weighted_score_scaler = StandardScaler()
        self.max_domain_scaler = StandardScaler()
        self.cls_proportion_scaler = StandardScaler()
        self.domains_visited_scaler = StandardScaler()
        self.psd_df_scaler = StandardScaler()
        self.mean_scores_scaler = StandardScaler()
        self.params = params
        self.engine = engine

    def get_probability_score(self, x_valid_domains):
        user_activity_timeseries = get_user_activity_timeseries(
            x_valid_domains, self.params["user_activity_timeseries"])

        p_scores_df = get_user_domain_scores(self.domain_activity,
                                             user_activity_timeseries)

        p_scores_df -= 0.5
        for col in self.valid_domains:
            if col not in p_scores_df.columns:
                p_scores_df[col] = np.nan
        p_scores_df = p_scores_df[self.valid_domains]
        # min_max_scale_all_values(p_scores_df, train_devices, self.score_scaler)
        return p_scores_df

    def get_cls_features(self, cls_data):
        cls_proportion = get_cls_proportion(cls_data)
        for col in self.cls_columns:
            if col not in cls_proportion.columns:
                cls_proportion[col] = np.nan
        cls_proportion = cls_proportion[self.cls_columns]
        z_normalize_by_all(cls_proportion,
                           train_devices=None,
                           per_column=True,
                           fill_na_pre_transform=True,
                           scaler=self.cls_proportion_scaler)
        return cls_proportion

    def get_content_based_features(self, x_valid_domains):
        domain_usage_proportion = get_domain_usage_proportion(x_valid_domains)
        max_domain_usage = domain_usage_proportion.max(axis=1).to_frame()
        z_normalize_by_all(df=max_domain_usage,
                           train_devices=None,
                           per_column=True,
                           scaler=self.max_domain_scaler)

        domain_usage_proportion = np.log(1 + domain_usage_proportion)
        domain_usage_proportion = ((domain_usage_proportion.T) /
                                   domain_usage_proportion.T.max()).T
        for col in self.valid_domains:
            if col not in domain_usage_proportion.columns:
                domain_usage_proportion[col] = np.nan
        domain_usage_proportion = domain_usage_proportion[self.valid_domains]
        return domain_usage_proportion, max_domain_usage

    def get_frequency_based_features(self, x, valid_x):
        psd_df = get_ps_df(x, self.engine, ntimebins=168)
        z_normalize_by_all(psd_df,
                           train_devices=None,
                           per_column=True,
                           scaler=self.psd_df_scaler)
        domains_visited_proportion = get_proportion_of_domains_visited(valid_x)
        for col in self.domains_visited_cols:
            if col not in domains_visited_proportion.columns:
                domains_visited_proportion[col] = np.nan
        domains_visited_proportion = domains_visited_proportion[
            self.domains_visited_cols]
        z_normalize_by_all(domains_visited_proportion,
                           train_devices=None,
                           per_column=True,
                           fillval=0,
                           scaler=self.domains_visited_scaler)
        return psd_df, domains_visited_proportion

    def get_mixed_features(self, p_scores_df, domain_usage_proportion):
        weighted_final_scores = get_weighted_final_scores(
            p_scores_df, domain_usage_proportion)
        z_normalize_by_all(weighted_final_scores,
                           train_devices=None,
                           per_column=False,
                           fill_na_pre_transform=False,
                           scaler=self.weighted_score_scaler)

        # add mean and std of weighted_final_scores
        return weighted_final_scores, weighted_final_scores.T.mean().to_frame(
        ), weighted_final_scores.T.std().to_frame()

    def load_standard_scaler(self, scaler_path):
        '''
        Load the StandardScaler from the given path
        '''
        with open(scaler_path, 'r') as f:
            loaded_params = json.load(f)
        scaler = StandardScaler()
        scaler.mean_ = np.array([loaded_params["mean_"]], dtype=np.float64)
        scaler.var_ = np.array([loaded_params["var_"]], dtype=np.float64)
        scaler.scale_ = np.array([loaded_params["scale_"]], dtype=np.float64)
        scaler.n_samples_seen_ = np.array([loaded_params["n_samples_seen_"]],
                                          dtype=np.int64)
        return scaler

    def load_minmax_scaler(self, scaler_path):
        '''
        Load the MinMaxScaler from the given path
        '''
        with open(scaler_path, 'r') as f:
            loaded_params = json.load(f)
        scaler = MinMaxScaler()
        scaler.min_ = np.array([loaded_params["min_"]], dtype=np.float64)
        scaler.scale_ = np.array([loaded_params["scale_"]], dtype=np.float64)
        scaler.data_min_ = np.array([loaded_params["data_min_"]],
                                    dtype=np.float64)
        scaler.data_max_ = np.array([loaded_params["data_max_"]],
                                    dtype=np.float64)
        scaler.data_range_ = np.array([loaded_params["data_range_"]],
                                      dtype=np.float64)
        scaler.feature_range = tuple(loaded_params["feature_range"])
        return scaler

    def load(self, dir_path):
        '''
        Load the trained model and domain activity data
        '''
        import os
        import json

        model_path = os.path.join(dir_path, 'XGB_model.json')
        self.model.load_model(model_path)
        best_features_path = os.path.join(dir_path, 'best_features.json')
        with open(best_features_path, "r") as fp:
            self.best_features = json.load(fp)
        self.best_domains = [
            int(x.replace("p_", "")) for x in self.best_features
            if x.startswith("p_") and "mean" not in x
        ]
        domain_activity_path = os.path.join(dir_path,
                                            'best_domains_timeseries.parquet')
        self.domain_activity = self.engine.read_parquet(domain_activity_path)
        # self.load_minmax_scaler(os.path.join(dir_path, 'minmax_scaler.json'))
        scalers = [
            "max_domain_scaler", "psd_df_scaler", "domains_visited_scaler",
            "cls_proportion_scaler", "weighted_score_scaler",
            "mean_scores_scaler"
        ]
        for s in scalers:
            setattr(
                self, s,
                self.load_standard_scaler(os.path.join(dir_path, f'{s}.json')))
        with open(os.path.join(dir_path, 'cls_columns.json'), 'r') as f:
            self.cls_columns = json.load(f)

        with open(os.path.join(dir_path, 'valid_domains.json'), 'r') as f:
            self.valid_domains = [int(x) for x in json.load(f)]
        with open(
                os.path.join(dir_path,
                             'domains_visited_proportions_cols.json'),
                'r') as f:
            self.domains_visited_cols = json.load(f)

    def prepare_data(self, X):
        X = X.copy()

        if X['Datetime'].dtype == 'O':
            X['Datetime'] = self.engine.to_datetime(X['Datetime'])
        if "Device_ID" not in X.columns:
            X["Device_ID"] = 1
        X.set_index(['Datetime'], inplace=True)
        x_valid_domains = X[X['Domain_Name'].isin(self.valid_domains)]
        return X, x_valid_domains

    def predict(self, X):
        '''
        Predict the class probability for the input data
        '''
        # Process user timeseries
        X, x_valid_domains = self.prepare_data(X)
        p_scores_df = self.get_probability_score(x_valid_domains)
        domain_usage_proportion, max_domain_usage = self.get_content_based_features(
            x_valid_domains)
        cls_proportion = self.get_cls_features(X)
        psd_df, domains_visited_proportion = self.get_frequency_based_features(
            X, valid_x=x_valid_domains)
        weighted_final_scores, mean_probability_score, std_probability_score = self.get_mixed_features(
            p_scores_df, domain_usage_proportion)
        final_features = join_features(
            device_targets=None,
            weighted_final_scores=weighted_final_scores,
            cls_proportion=cls_proportion,
            psd_df=psd_df,
            domains_visited_proportion=domains_visited_proportion,
            mean_probability_score=mean_probability_score,
            max_domain_usage=max_domain_usage)
        final_features = final_features[self.best_features]
        # add feature that is number of zeros in final_features

        # Make prediction
        prediction = self.model.predict(final_features)
        return prediction[0]
