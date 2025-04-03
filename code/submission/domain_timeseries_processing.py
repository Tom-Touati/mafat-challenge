import json
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from functools import partial
import ray

def process_activity_timeseries(domain_df, bin_hours=6, gaussian_filter=True, n_days_each_side=3, std=1.5, drop_na=True, drop_zeros=False):
    activity_per_3h = domain_df[["Device_ID"]].resample(
        f'{str(bin_hours)}h').nunique()
    activity_per_3h.rename(columns={"Device_ID": "Activity"}, inplace=True)
    # activity_per_3h = activity_per_3h.to_frame()

    # n_days_each_side * 24h / 3h_per_bin * 2 sides
    gaussian_window_hours = int(n_days_each_side*24/bin_hours*2)
    if gaussian_filter:
        activity_per_3h = activity_per_3h.rolling(
            window=gaussian_window_hours, win_type='gaussian', center=True, min_periods=1, closed="both").mean(std=std)
    if drop_na:
        activity_per_3h.dropna(inplace=True)
    if drop_zeros:
        activity_per_3h = activity_per_3h[activity_per_3h["Activity"] != 0]
    return activity_per_3h.round().astype(int)


def _calculate_p1_if_active(domain_activity_timeseries):
    domain_activity_timeseries["bin_activity"] = domain_activity_timeseries["Activity_0"] + \
        domain_activity_timeseries["Activity_1"]
    domain_activity_timeseries["total_domain_activity"] = (
        domain_activity_timeseries["target_domain_activity_0"]+domain_activity_timeseries["target_domain_activity_1"])
    domain_activity_timeseries["relative_bin_activity"] = domain_activity_timeseries["bin_activity"] / \
        domain_activity_timeseries["total_domain_activity"]
    domain_activity_timeseries["p_Active|0"] = domain_activity_timeseries["Activity_0"]/(
        domain_activity_timeseries["0_users"]
    )
    domain_activity_timeseries["p_active|1"] = domain_activity_timeseries["Activity_1"]/(
        domain_activity_timeseries["1_users"])
    domain_activity_timeseries["p_1"] = domain_activity_timeseries["1_users"]/(
        domain_activity_timeseries["0_users"]+domain_activity_timeseries["1_users"])
    domain_activity_timeseries["p_active"] = domain_activity_timeseries["bin_activity"]/(
        domain_activity_timeseries["0_users"]+domain_activity_timeseries["1_users"])
    domain_activity_timeseries["p_1|active"] = domain_activity_timeseries["p_active|1"]*(
        domain_activity_timeseries["p_1"]/domain_activity_timeseries["p_active"])
    pass


def get_domain_activity_timeseries(train_df, domain_ts_kwargs):
    process_domain_timeseries = partial(
        process_activity_timeseries, **domain_ts_kwargs)
    process_domain_timeseries.__name__ = process_activity_timeseries.__name__
    domain_timeseries = train_df[["Domain_Name", "Target", "Device_ID"]].groupby(
        ["Domain_Name", "Target"]).apply(process_domain_timeseries)
    domain_timeseries["activity_fraction"] = domain_timeseries.groupby(
        ["Domain_Name", "Target"]).transform(lambda x: x/x.sum())
    # Add the sum of activity as a new column
    domain_activity = domain_timeseries.groupby(
        ["Domain_Name", "Target"])[["Activity"]].sum()
    domain_activity = domain_activity.rename(
        columns={"Activity": "target_domain_activity"})
    # Merge the results
    domain_timeseries = domain_timeseries.merge(
        domain_activity, left_index=True, right_index=True)
    # Reset index to get Target as a column, then pivot to get Target as columns
    pivot_fraction_ts = domain_timeseries.reset_index().pivot(
        index=['Datetime', 'Domain_Name'],
        columns='Target'
    ).fillna(0)
    pivot_fraction_ts.columns = [
        f'{col[0]}_{col[1]}' for col in pivot_fraction_ts.columns]
    target_users_per_domain = train_df.groupby(["Domain_Name", "Target"])["Device_ID"].nunique(
    ).unstack().fillna(0).astype(int).rename(columns={0: "0_users", 1: "1_users"})
    pivot_fraction_ts = pivot_fraction_ts.reset_index("Datetime").join(
        target_users_per_domain, how="left").set_index("Datetime", append=True).swaplevel()
    _calculate_p1_if_active(pivot_fraction_ts)
    return pivot_fraction_ts


def get_user_activity_timeseries(db_df, user_ts_kwargs):
    process_user_timeseries = partial(
        process_activity_timeseries, **user_ts_kwargs)
    process_user_timeseries.__name__ = process_activity_timeseries.__name__
    user_timeseries = db_df[["Domain_Name", "Device_ID"]].groupby(
        ["Domain_Name", "Device_ID"]).apply(process_user_timeseries)
    return user_timeseries


def get_user_domain_scores(domain_activity_timeseries, user_activity_timeseries):
    # Merge domain and user timeseries data more efficiently
    merged_timeseries_df = domain_activity_timeseries[["p_1|active", "bin_activity"]].reset_index().merge(
        user_activity_timeseries.reset_index(),
        how="left",
        on=["Domain_Name", "Datetime"]
    ).set_index(["Datetime", "Domain_Name", "Device_ID"])

    # Filter for active periods first to reduce data size
    merged_timeseries_df = merged_timeseries_df[merged_timeseries_df["Activity"] > 0]

    # Calculate scores directly
    # Calculate relative activity using transform for vectorized operation
    group_sums = merged_timeseries_df.groupby(['Domain_Name', 'Device_ID'])[
        'bin_activity'].transform('sum')

    # Vectorized division
    merged_timeseries_df['relative_active_bins_activity'] = merged_timeseries_df['bin_activity'] / group_sums

    # Calculate weighted scores in one step
    merged_timeseries_df["weighted_score"] = (merged_timeseries_df["p_1|active"]) * \
        (merged_timeseries_df["relative_active_bins_activity"])

    # Get final scores with optimized groupby
    final_scores = merged_timeseries_df.groupby(
        ["Device_ID", "Domain_Name"])["weighted_score"].sum()

    # Create pivot table efficiently
    final_scores_pivot = final_scores.unstack()

    return final_scores_pivot


import matplotlib.pyplot as plt
import numpy as np
def get_weighted_final_scores(final_scores_pivot,domain_usage_proportion,square_usage=False):
    # plt.figure()
    mult = domain_usage_proportion[final_scores_pivot.columns
        ]
    if square_usage:
        mult = mult**2
    weighted_final_scores = final_scores_pivot.mul(mult)

    # target_features = device_targets.set_index("Device_ID").join(weighted_final_scores)
    # target_features.set_index("Target")[[int(x) for x in best_features if x.isnumeric()]].stack().groupby("Target").mean().plot(kind="bar")
    # plt.show()
    return weighted_final_scores
    # with open('submission/minmax_scaler.json', 'w') as f:
    #     json.dump(scaler_params, f)
