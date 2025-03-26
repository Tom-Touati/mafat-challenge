
import xgboost
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import json
class model:
    def __init__(self,with_neptune=False):
        '''
        Init the model
        '''
        self.model = xgboost.XGBRegressor(
            seed=0, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            learning_rate=0.1,
            n_estimators=150, 
            max_depth=6, 
            objective='binary:logistic',
            eval_metric=roc_auc_score
        )
        self.domain_activity = None
        self.user_activity = None
        self.best_features = None
        self.score_scaler = MinMaxScaler(feature_range=(-1,1))
    def get_cls_proportion(self,df):
        cols=["Domain_cls1","Domain_cls2","Domain_cls3","Domain_cls4"]
        df = df[cols]
        df = df.stack()
        df = df[df!=0]
        df = df.value_counts(normalize=True).T.fillna(0)
        df = df.to_frame().T
        df.columns = [f"cls_{col}" for col in df.columns]
        missing_cols = [x for x in self.best_features if x not in df.columns and x.startswith("cls_")]
        if len(missing_cols)>0:
            df[missing_cols] = np.zeros((df.shape[0], len(missing_cols)))
        return df
    def process_activity_timeseries(self,domain_df,bin_hours=6,gaussian_filter=True,n_days_each_side=3,std=1.5,drop_na=True,drop_zeros=False):
        activity_per_3h = domain_df[["Device_ID"]].resample(f'{str(bin_hours)}h').nunique()
        activity_per_3h.rename(columns={"Device_ID":"Activity"},inplace=True)
        # activity_per_3h = activity_per_3h.to_frame()

        gaussian_window_hours = int(n_days_each_side*24/bin_hours*2) # n_days_each_side * 24h / 3h_per_bin * 2 sides
        if gaussian_filter:
            activity_per_3h = activity_per_3h.rolling(window=gaussian_window_hours, win_type='gaussian',center=True,min_periods=1,closed="both").mean(std=std)
        if drop_na:
            activity_per_3h.dropna(inplace=True)
        if drop_zeros:
            activity_per_3h = activity_per_3h[activity_per_3h["Activity"]!=0]
        return activity_per_3h.round().astype(int)

    def class_probability_score(self,active, p_active_given_a, p_active_given_b, prior_a=0.5, total_users=100):
        """
        Calculate class probability score with vectorized operations
        
        Args:
            active: Boolean indicating if user was active
            p_active_given_a: Probability of activity given class A (0)
            p_active_given_b: Probability of activity given class B (1)
            prior_a: Prior probability for class A
            total_users: Total number of users for confidence calculation
        """
        # Use numpy for vectorized operations
        likelihood_a = np.where(active, p_active_given_a, 1 - p_active_given_a)
        likelihood_b = np.where(active, p_active_given_b, 1 - p_active_given_b)
        
        # Avoid division by zero
        evidence = (likelihood_a * prior_a + likelihood_b * (1 - prior_a))
        posterior_a = (likelihood_a * prior_a) / evidence

        return posterior_a #* confidence_factor


    def get_user_domain_scores(self,user_activity_timeseries,domain_activity_timeseries):
        merged_timeseries_df = domain_activity_timeseries.reset_index().merge(
            user_activity_timeseries.reset_index(), how="inner", on=["Domain_Name", "Datetime"]
        ).set_index(["Datetime", "Domain_Name", "Device_ID"])

        merged_timeseries_df["bin_activity"] = merged_timeseries_df["Activity_0"]+merged_timeseries_df["Activity_1"]
        merged_timeseries_df["total_activity"] = (merged_timeseries_df["target_domain_activity_0"]+merged_timeseries_df["target_domain_activity_1"])
        merged_timeseries_df["relative_0_activity"] = merged_timeseries_df["target_domain_activity_0"]/merged_timeseries_df["total_activity"]
        merged_timeseries_df["score"]=self.class_probability_score(merged_timeseries_df["Activity"], merged_timeseries_df["activity_fraction_0"], merged_timeseries_df["activity_fraction_1"], 
                                                                   prior_a=merged_timeseries_df["relative_0_activity"], total_users=merged_timeseries_df["bin_activity"])
        merged_timeseries_df["relative_bin_activity"] = merged_timeseries_df["bin_activity"]/merged_timeseries_df["total_activity"]
        merged_timeseries_df["weighted_score"] = (merged_timeseries_df["score"])*(merged_timeseries_df["bin_activity"])
        final_scores = merged_timeseries_df.groupby(["Device_ID","Domain_Name"])["weighted_score"].mean()
        final_scores_pivot = final_scores.to_frame().reset_index().pivot(index="Device_ID",columns="Domain_Name").fillna(0)
        final_scores_pivot = final_scores_pivot.droplevel(0, axis=1)
        final_scores_pivot.columns = [str(col) for col in final_scores_pivot.columns]
        missing_columns = [x for x in self.best_domains if x not in final_scores_pivot.columns]
        final_scores_pivot[missing_columns] = np.zeros((final_scores_pivot.shape[0], len(missing_columns)))
        final_scores_pivot = final_scores_pivot[self.best_domains]
        return final_scores_pivot
    def load_minmax_scaler(self,scaler_path):
        '''
        Load the MinMaxScaler from the given path
        '''
        with open(scaler_path, 'r') as f:
            loaded_params = json.load(f)
        self.score_scaler.min_ = np.array([loaded_params["min_"]], dtype=np.float64)
        self.score_scaler.scale_ = np.array([loaded_params["scale_"]], dtype=np.float64)
        self.score_scaler.data_min_ = np.array([loaded_params["data_min_"]], dtype=np.float64)
        self.score_scaler.data_max_ = np.array([loaded_params["data_max_"]], dtype=np.float64)
        self.score_scaler.data_range_ = np.array([loaded_params["data_range_"]], dtype=np.float64)
        self.score_scaler.feature_range = tuple(loaded_params["feature_range"])
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
        self.best_domains = [x for x in self.best_features if x.isnumeric()]
        domain_activity_path = os.path.join(dir_path, 'best_domain_activity.parquet')
        self.domain_activity = pd.read_parquet(domain_activity_path)
        self.load_minmax_scaler(os.path.join(dir_path,'minmax_scaler.json'))
    def predict(self, X):
        '''
        Predict the class probability for the input data
        '''
        # Process user timeseries
        X = X.copy()
        
        if X['Datetime'].dtype == 'O':
            X['Datetime'] = pd.to_datetime(X['Datetime'])
        if "Device_ID" not in X.columns:
            X["Device_ID"] = 1
        X.set_index(['Datetime'], inplace=True)
        x_domains = X[X['Domain_Name'].isin([int(x) for x in self.best_domains])]
        user_timeseries = x_domains[['Device_ID', 'Domain_Name']].groupby(["Device_ID","Domain_Name"]).apply(lambda x :self.process_activity_timeseries(
            x,
            bin_hours=6,
            gaussian_filter=True,
            n_days_each_side=3,
            std=1.5,
            drop_na=True,
            drop_zeros=False
        ))
        
        # Get domain scores
        final_scores = self.get_user_domain_scores(user_timeseries, self.domain_activity)
        final_scores.iloc[:] = self.score_scaler.transform(final_scores.values.reshape(-1,1)).reshape(final_scores.shape)
        cls_proportion = self.get_cls_proportion(X)
        final_features = pd.concat([final_scores.reset_index(),cls_proportion.reset_index()],axis=1)[self.best_features]
        # Make prediction
        prediction = self.model.predict(final_features)
        return prediction[0]
