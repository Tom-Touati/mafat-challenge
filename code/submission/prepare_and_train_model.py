# model training
import xgboost
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score


def join_features(device_targets, weighted_final_scores, domain_usage_proportion, cls_proportion, psd_df,domains_visited_proportion):
    final_features = device_targets.set_index(
        "Device_ID").join(weighted_final_scores.rename(columns = lambda x: "p_"+str(x)
            ), how="left")
    final_features = final_features.join(domain_usage_proportion.rename(columns = lambda x: "domain_usage_"+str(x)
            ), how="left")
    final_features = final_features.join(cls_proportion.rename(columns = lambda x: "cls_proportion_"+str(x)
            ), how="left")
    final_features = final_features.join(psd_df.rename(columns = lambda x: "activity_ps_"+str(x)
            ), how="left")
    final_features = final_features.join(domains_visited_proportion.rename(columns = lambda x: "domains_visited_"+str(x)
            ), how="left")
    # final_features = final_features.join(device_domain_PCA,how="left")
    # if active_days is not None:
    #     final_features = final_features.join(active_days,how="left")
    # if activity_per_time_range is not None:
    #     final_features = final_features.join(activity_per_time_range,how="left")
    # final_features = final_features.fillna(0)
    final_features.columns = [str(col) for col in final_features.columns]
    return final_features


def prepare_model_data(final_features, train_devices, test_device_ids):
    X_train = final_features[final_features.index.isin(
        train_devices)].drop('Target', axis=1)
    y_train = final_features[final_features.index.isin(
        train_devices)]['Target']

    X_test = final_features[final_features.index.isin(
        test_device_ids)].drop('Target', axis=1)
    y_test = final_features[final_features.index.isin(
        test_device_ids)]['Target']
    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train, X_test=None, y_test=None, params=None):
    deval = xgboost.DMatrix(X_test, y_test) if X_test is not None else None

    xgb_reg = xgboost.XGBRegressor(**params, eval_metric=roc_auc_score)#,eval_set=[deval,"eval"],verbose_eval=True)
    selector = RFE(xgb_reg, n_features_to_select=1000, step=20000)
    selector = selector.fit(X_train, y_train)
    best_features = list(X_train.columns[selector.support_])
    if X_test is not None:
        test_prediction = selector.estimator_.predict(X_test[best_features])
        test_auc = round(roc_auc_score(y_test, test_prediction), 3)
        return test_auc, selector, best_features, y_test, test_prediction
    return None, selector, best_features, None, None
