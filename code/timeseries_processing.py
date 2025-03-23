# preprocess timeseries
from functools import partial
def process_activity_timeseries(domain_df,bin_hours=6,gaussian_filter=True,n_days_each_side=3,std=1.5,drop_na=True,drop_zeros=False):
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

def get_domain_activity_timeseries(train_df,domain_ts_kwargs):
    process_domain_timeseries = partial(process_activity_timeseries,**domain_ts_kwargs)
    process_domain_timeseries.__name__ =process_activity_timeseries.__name__
    domain_timeseries = train_df[["Domain_Name","Target","Device_ID"]].groupby(["Domain_Name","Target"]).apply(process_domain_timeseries)
    
    domain_timeseries = domain_timeseries
    domain_timeseries["activity_fraction"] = domain_timeseries.groupby(["Domain_Name", "Target"]).transform(lambda x: x/x.sum())
    # Add the sum of activity as a new column
    domain_activity = domain_timeseries.groupby(["Domain_Name", "Target"])[["Activity"]].sum()

    domain_activity = domain_activity.rename(columns={"Activity": "target_domain_activity"})
    # Merge the results
    domain_timeseries = domain_timeseries.merge(domain_activity, left_index=True, right_index=True)
    # Reset index to get Target as a column, then pivot to get Target as columns
    pivot_fraction_ts = domain_timeseries.reset_index().pivot(
        index=['Datetime', 'Domain_Name'],
        columns='Target'
    ).fillna(0)
    pivot_fraction_ts.columns = [f'{col[0]}_{col[1]}' for col in pivot_fraction_ts.columns]
    return pivot_fraction_ts
def get_user_activity_timeseries(db_df,user_ts_kwargs):
    process_user_timeseries = partial(process_activity_timeseries,**user_ts_kwargs)
    process_user_timeseries.__name__ =process_activity_timeseries.__name__
    user_timeseries = db_df[["Domain_Name","Device_ID"]].groupby(["Domain_Name","Device_ID"]).apply(process_user_timeseries)
    # user_timeseries.index = mpd.MultiIndex.from_tuples(user_timeseries.index.to_list(),names=["Domain_Name","Device_ID","Datetime"])
    return user_timeseries
