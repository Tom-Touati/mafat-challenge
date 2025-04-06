# model training
import xgboost
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score


def join_features(
    device_targets,
    weighted_final_scores=None,
    domain_usage_proportion=None,
    cls_proportion=None,
    psd_df=None,
    domains_visited_proportion=None,
    mean_probability_score=None,
    max_domain_usage=None,
    user_url_score=None,
    cls_final_scores=None,
):
    final_features = device_targets.set_index(
        "Device_ID") if device_targets is not None else None

    if weighted_final_scores is not None and final_features is not None:
        final_features = final_features.join(
            weighted_final_scores.rename(columns=lambda x: "p_" + str(x)),
            how="left")
        print("weighted_final_scores")
        print(weighted_final_scores.stack().describe())
    else:
        final_features = weighted_final_scores.rename(
            columns=lambda x: "p_" + str(x))

    if domain_usage_proportion is not None:
        final_features = final_features.join(domain_usage_proportion.rename(
            columns=lambda x: "domain_usage_" + str(x)),
                                             how="left")
        print("domain_usage_proportion")
        print(domain_usage_proportion.stack().describe())
    if cls_proportion is not None:
        final_features = final_features.join(cls_proportion.rename(
            columns=lambda x: "cls_proportion_" + str(x)),
                                             how="left")
        print("cls_proportion")
        print(cls_proportion.stack().describe())
    if psd_df is not None:
        final_features = final_features.join(
            psd_df.rename(columns=lambda x: "activity_ps_" + str(x)),
            how="left")
        print("psd_df")
        print(psd_df.stack().describe())
    if domains_visited_proportion is not None:
        final_features = final_features.join(domains_visited_proportion.rename(
            columns=lambda x: "domains_visited_" + str(x)),
                                             how="left")
        print("domains_visited_proportion")
        print(domains_visited_proportion.stack().describe())
    if mean_probability_score is not None:
        final_features = final_features.join(mean_probability_score.rename(
            columns=lambda x: "mean_p_" + str(x)),
                                             how="left")
        print("mean_probability_score")
        print(mean_probability_score.stack().describe())
    if max_domain_usage is not None:
        final_features = final_features.join(max_domain_usage.rename(
            columns=lambda x: "max_domain_usage_" + str(x)),
                                             how="left")
        print("max_domain_usage")
        print(max_domain_usage.stack().describe())
    if user_url_score is not None:
        final_features = final_features.join(
            user_url_score.rename(columns=lambda x: "user_url_" + str(x)),
            how="left")
        print("user_url_score")
        print(user_url_score.stack().describe())
    if cls_final_scores is not None:
        final_features = final_features.join(cls_final_scores.rename(
            columns=lambda x: "cls_final_scores_" + str(x)),
                                             how="left")
    # final_features = final_features.join(device_domain_PCA,how="left")
    # if active_days is not None:
    #     final_features = final_features.join(active_days,how="left")
    # if activity_per_time_range is not None:
    #     final_features = final_features.join(activity_per_time_range,how="left")
    # final_features = final_features.fillna(0)
    final_features.columns = [str(col) for col in final_features.columns]
    return final_features


def prepare_model_data(final_features, train_devices, test_device_ids):
    X_train = final_features[final_features.index.isin(train_devices)].drop(
        'Target', axis=1)
    y_train = final_features[final_features.index.isin(
        train_devices)]['Target']

    # X_test = final_features[final_features.index.isin(test_device_ids)].drop(
        # 'Target', axis=1)
    # y_test = final_features[final_features.index.isin(
        # test_device_ids)]['Target']
    return X_train, y_train#, X_test[X_train.columns], y_test


def train_model(X_train, y_train, X_test=None, y_test=None, params=None):

    xgb_reg = xgboost.XGBRegressor(
        **params["model"],
        eval_metric=roc_auc_score)  #,early_stopping_rounds=25)
    selector = RFE(xgb_reg, **params["feature_selection"], verbose=1)
    selector = selector.fit(
        X_train,
        y_train)  #, eval_set=[(X_train,y_train),(X_test, y_test)], verbose=1)
    best_features = list(X_train.columns[selector.support_])
    if X_test is not None:
        test_prediction = selector.estimator_.predict(X_test[best_features])
        test_auc = round(roc_auc_score(y_test, test_prediction), 3)
        return test_auc, selector, best_features, y_test, test_prediction
    return selector, best_features


def train_without_selection(X_train,
                            y_train,
                            X_test=None,
                            y_test=None,
                            params=None):
    xgb_reg = xgboost.XGBRegressor(
        **params["model"],
        eval_metric=roc_auc_score)  #,early_stopping_rounds=25)
    xgb_reg.fit(X_train, y_train)
    if X_test is not None:
        test_prediction = xgb_reg.predict(X_test)
        test_auc = round(roc_auc_score(y_test, test_prediction), 3)
        return test_auc, xgb_reg
    return None, xgb_reg
