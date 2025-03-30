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
def load_data_from_db(con,domain_cls=False,only_domain=False):
    try:
        additional_query = ""
        if domain_cls:
            additional_query = ",Domain_cls1,Domain_cls2,Domain_cls3,Domain_cls4"
        device_ids_query = f"""SELECT Domain_Name, Device_ID, Target,Datetime{additional_query} from data"""
        
        if only_domain:
            device_ids_query = f"""SELECT Domain_Name{additional_query} from data"""
        # WHERE Domain_Name != 1732927 """
        df = mpd.read_sql(device_ids_query, con
                         )._repartition()
        df = df[df['Domain_Name'] != 1732927]
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def load_and_prepare_data():
    freeze_support()
    dbfile = '../../data/training_set.db'

    conn = ModinDatabaseConnection('sqlalchemy', f'sqlite:///{dbfile}')

    # Can use get_connection to get underlying sqlalchemy engine
    conn.get_connection()
    db_df = load_data_from_db(conn)
    print(db_df.head())
    del conn
    db_df['Datetime'] = mpd.to_datetime(db_df['Datetime'])
    db_df.set_index('Datetime', inplace=True)
    db_df = db_df.astype( {'Domain_Name': 'uint32', 'Device_ID': 'uint32', 'Target': 'uint8'})
    return db_df
def get_cls_data():
    """
    Feature engeinering: For each Device_ID calculate the proportions of all the domain_cls he entered.

    Parameters
    ----------
    conn: connection
        A connection object to the database.
    device_list : list
        A list of Device_IDs to calculate their proportions.

    Returns
    -------
    dataframe
        A dataframe with the proportions for each Device_IDs and Domain_cls.
    """
    freeze_support()
    dbfile = '../../data/training_set.db'

    conn = ModinDatabaseConnection('sqlalchemy', f'sqlite:///{dbfile}')

    # Can use get_connection to get underlying sqlalchemy engine
    conn.get_connection()
    sql = '''SELECT
                    Device_ID,
                    Domain_Name,
                    Domain_cls1,
                    Domain_cls2,
                    Domain_cls3,
                    Domain_cls4
                    from data
                    where Domain_Name != 1732927 and Domain_cls1 != 0 and Domain_cls2 != 0 and Domain_cls3 != 0 and Domain_cls4 != 0
                    '''

    df = mpd.read_sql(sql,conn, columns = ['Device_ID','Domain_cls', 'proportion'])
    
    return df
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
