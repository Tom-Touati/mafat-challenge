import os
import sqlite3
# %matplotlib widget
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
from modin.db_conn import ModinDatabaseConnection
import modin.pandas as mpd
  # Modin will use Ray
# ray.init()
# NPartitions.put(16)
def load_domain_data_from_db(con,domain_cls=False,only_domain=False):
    try:
        device_ids_query = f"""SELECT Datetime,Device_ID,Domain_Name,Target from data
        WHERE Domain_Name != 1732927
        """
        
        # WHERE Domain_Name != 1732927 """
        df = mpd.read_sql(device_ids_query, con
                         )._repartition()
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
def load_cls_data_from_db(con,domain_cls=False,only_domain=False):
    try:
        device_ids_query = f"""SELECT Datetime,Device_ID,Domain_cls1,Domain_cls2,Domain_cls3,Domain_cls4 from data
        from data
        WHERE Domain_Name != 1732927
        """
        
        # WHERE Domain_Name != 1732927 """
        df = mpd.read_sql(device_ids_query, con
                         )._repartition()
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
def load_and_prepare_data(data_type="domain"):
    freeze_support()
    dbfile = '../../data/training_set.db'

    conn = ModinDatabaseConnection('sqlalchemy', f'sqlite:///{dbfile}')

    # Can use get_connection to get underlying sqlalchemy engine
    conn.get_connection()
    if data_type=="domain":
        db_df = load_domain_data_from_db(conn)
    elif data_type=="cls":
        db_df = load_cls_data_from_db(conn)
    
    print(db_df.head())
    del conn
    db_df['Datetime'] = mpd.to_datetime(db_df['Datetime'])
    db_df.set_index('Datetime', inplace=True)
    if data_type=="domain":
        db_df = db_df.astype( {'Domain_Name': 'uint32', 'Device_ID': 'uint32', 'Target': 'category'})
    elif data_type=="cls":
        db_df = db_df.astype( {'Device_ID': 'uint32','Domain_cls1': 'category', 'Domain_cls2': 'category', 'Domain_cls3': 'category', 'Domain_cls4': 'category','Target': 'category'})
    return db_df

# prepare training data
import matplotlib.pyplot as plt
#train test split
from sklearn.model_selection import train_test_split
def get_train_test_devices(device_target_df, test_size=0.2, random_state=43):    
    # Perform stratified split on device IDs
    train_device_ids, test_device_ids = train_test_split(
        device_target_df['Device_ID'],
        test_size=test_size,
        random_state=random_state,
        stratify=device_target_df['Target']
    )
    return train_device_ids, test_device_ids

def get_initial_train_data(db_df, test_size=0.2, random_state=42, min_domain_devices=10,n_devices_hist=False):
    device_targets = db_df.groupby("Device_ID")["Target"].first().reset_index()
    train_devices, test_device_ids = get_train_test_devices(device_targets,test_size=test_size,random_state=random_state)
    train_df = db_df[db_df["Device_ID"].isin(train_devices)]
    devices_per_domain = train_df.groupby("Domain_Name")["Device_ID"].nunique()
    
    domain_mask = devices_per_domain>min_domain_devices
    print(f"Percentage of domains with more than {min_domain_devices} devices: {domain_mask.mean()*100:.2f}%")
    devices_per_domain = devices_per_domain[domain_mask]
    if n_devices_hist:
        hist = devices_per_domain.hist()
        # run["plots/domain_devices_hist"].upload(neptune.types.File.as_image(hist.figure))
        plt.show()
    train_df = train_df[train_df["Domain_Name"].isin(devices_per_domain.index)]
    return train_df,train_devices,test_device_ids, device_targets, devices_per_domain

# Add this line
