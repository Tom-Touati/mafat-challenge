
import xgboost
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

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
            objective='binary:logistic'
        )
        self.domain_activity = None
        self.user_activity = None
        self.best_features = None
        if with_neptune:
            self.init_neptune()

    def init_neptune(self,tags=["time-based-models", "activity-based-features"]):
        self.run = neptune.init(
                project="tom.touati/web-segmentation",  # replace with your project
                api_token=os.environ["NEPTUNE_API_TOKEN"],
                capture_stdout=True,
                capture_stderr=True,
                capture_hardware_metrics=True,
                tags=tags,
                description="User activity patterns analysis",
                mode=NEPTUNE_MODE
            )
    def process_activity_timeseries(self, domain_df, bin_hours=6, gaussian_filter=True, 
                                  n_days_each_side=3, std=1.5, drop_na=True, drop_zeros=False):
        activity_per_3h = domain_df[["Device_ID"]].resample(f'{str(bin_hours)}h').nunique()
        activity_per_3h.rename(columns={"Device_ID":"Activity"}, inplace=True)

        gaussian_window_hours = int(n_days_each_side*24/bin_hours*2)
        if gaussian_filter:
            activity_per_3h = activity_per_3h.rolling(
                window=gaussian_window_hours, 
                win_type='gaussian',
                center=True,
                min_periods=1,
                closed="both"
            ).mean(std=std)
            
        if drop_na:
            activity_per_3h.dropna(inplace=True)
        if drop_zeros:
            activity_per_3h = activity_per_3h[activity_per_3h["Activity"]!=0]
            
        return activity_per_3h.round().astype(int)

    def class_probability_score(self, active, p_active_given_a, p_active_given_b, prior_a=0.5):
        likelihood_a = np.where(active, p_active_given_a, 1 - p_active_given_a)
        likelihood_b = np.where(active, p_active_given_b, 1 - p_active_given_b)
        
        evidence = (likelihood_a * prior_a + likelihood_b * (1 - prior_a))
        posterior_a = (likelihood_a * prior_a) / evidence

        return posterior_a

    def get_user_domain_scores(self, user_timeseries, domain_activity):
        merged_df = domain_activity.reset_index().merge(
            user_timeseries.reset_index(), 
            how="inner", 
            on=["Domain_Name", "Datetime"]
        ).set_index(["Datetime", "Domain_Name", "Device_ID"])

        merged_df["bin_activity"] = merged_df["Activity_0"] + merged_df["Activity_1"]
        merged_df["total_activity"] = (merged_df["target_domain_activity_0"] + merged_df["target_domain_activity_1"])
        merged_df["relative_0_activity"] = merged_df["target_domain_activity_0"]/merged_df["total_activity"]
        
        merged_df["score"] = self.class_probability_score(
            merged_df["Activity"], 
            merged_df["activity_fraction_0"], 
            merged_df["activity_fraction_1"], 
            prior_a=merged_df["relative_0_activity"]
        )
        
        merged_df["weighted_score"] = merged_df["score"] * merged_df["bin_activity"]
        final_scores = merged_df.groupby(["Device_ID","Domain_Name"])["weighted_score"].mean()
        final_scores_pivot = final_scores.to_frame().reset_index().pivot(
            index="Device_ID",
            columns="Domain_Name"
        ).fillna(0)

        final_scores_pivot = (final_scores_pivot-final_scores_pivot.values.min())/(
            final_scores_pivot.values.max()-final_scores_pivot.values.min())*2-1
            
        return final_scores_pivot

    def load(self, dir_path):
        '''
        Load the trained model and domain activity data
        '''
        import os
        import json
        model_path = os.path.join(dir_path, 'XGB_model.json')
        self.model.load_model(model_path)
        best_features_path = os.path.join(dir_path, 'selected_features.json')
        with open(best_features_path, "r") as fp:
            self.best_features = json.load(fp)
        domain_activity_path = os.path.join(dir_path, 'best_domain_activity.parquet')
        self.domain_activity = pd.read_parquet(domain_activity_path)

    def predict(self, X):
        '''
        Predict the class probability for the input data
        '''
        # Process user timeseries
        X = X.copy()
        X = X[X['Domain_Name'].isin([int(x) for x in self.best_features)]
        X['Datetime'] = pd.to_datetime(X['Datetime'])
        X["Device_ID"] = 1
        X.set_index(['Datetime'], inplace=True)
        
        user_timeseries = X[['Device_ID', 'Domain_Name']].groupby(["Device_ID","Domain_Name"]).apply(lambda x :self.process_activity_timeseries(
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
        
        # Make prediction
        prediction = self.model.predict(final_scores)
        return prediction[0]
