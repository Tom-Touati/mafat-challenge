import numpy as np


def get_domain_usage_proportion(db_df):
    res = db_df.groupby("Device_ID")["Domain_Name"].value_counts(
        normalize=True).unstack(fill_value=0).astype(np.float32)
    return res


def get_proportion_of_domains_visited(df):
    res = df.groupby(
        "Device_ID")["Domain_Name"].nunique() / df["Domain_Name"].nunique()
    res.name = "n_domains"
    return res.to_frame()
