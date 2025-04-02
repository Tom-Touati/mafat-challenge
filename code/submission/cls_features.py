import numpy as np
def get_cls_proportion(df):
    cols = ["Domain_cls1","Domain_cls2","Domain_cls3","Domain_cls4"]
    df = df.set_index("Device_ID")[cols].stack()
    df = df[df!=0]
    df.index = df.index.droplevel(1)
    df = df.groupby("Device_ID").value_counts(normalize=True).unstack(fill_value=0).astype(np.float32)
    return df
