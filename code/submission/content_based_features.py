import numpy as np
def get_domain_usage_proportion(db_df):
    res =  db_df.groupby("Device_ID")["Domain_Name"].value_counts(normalize=True).unstack(fill_value=0).astype(np.float32)
    return res

def get_cls_proportion(df):
    cols = ["Domain_cls1","Domain_cls2","Domain_cls3","Domain_cls4"]
    df = df.groupby("Device_ID")[cols].apply(lambda x: df.stack().value_counts(normalize=True).T.fillna(0).to_frame().T)

    
    df.columns = [f"cls_{col}" for col in df.columns]
    df.rename({"cls_Device_ID":"Device_ID"},axis=1,inplace=True)
    print("check if device id is in columns", "Device_ID" in df.columns)
    df.index = df.index.droplevel(-1)
    return df
def get_proportion_of_domains_visited(df):
    res = df.groupby("Device_ID")["Domain_Name"].nunique()/df["Domain_Name"].nunique()
    res.name = "n_domains"
    return res.to_frame()

