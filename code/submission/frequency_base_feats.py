import numpy as np
import modin.pandas as mpd
from sklearn.decomposition import PCA

def get_active_days_per_user(user_domain_ts):
    """
    Calculate the number of unique days each user had any activity.

    Args:
        user_domain_ts: MultiIndex Series with levels [Domain_Name, Device_ID, Datetime]

    Returns:
        Series with index Device_ID and values being number of unique active days
    """
    # Reset index to get Datetime as a column
    df = user_domain_ts.reset_index()

    # Convert Datetime to date (removing time component)
    df['Date'] = df['Datetime'].dt.date

    # Group by Device_ID and count unique dates where Activity > 0
    active_days = df[df['Activity'] > 0].groupby('Device_ID')['Date'].nunique()

    active_days = active_days.astype(int)
    active_days.name = "Active_Days"
    active_days = (active_days-active_days.min()
                   ) / (active_days.max()-active_days.min())*2-1
    return active_days


def get_activity_per_time_bin(df, bin_hours=3):
    # Convert datetime to time only
    # time_index = db_df.index.to_series().dt.time
    # df["time"] = time_index
    df_copy = df.copy()
    df_copy["time"] = db_df.index.to_series().dt.hour.astype(int)//bin_hours
    df_copy["day_part_activity"] = 0
    activity_per_time_range = df_copy[["Device_ID", "time", "day_part_activity"]].groupby(
        ["Device_ID", "time"]).count()
    activity_per_time_range["activity_fraction"] = activity_per_time_range.groupby("Device_ID").apply(lambda x: x/x.sum()).values
    activity_per_time_range = activity_per_time_range[["activity_fraction"]].reset_index()
    activity_per_time_range = activity_per_time_range.pivot(index="Device_ID",columns="time",values="activity_fraction")
    activity_per_time_range.columns = [f"time_{col}" for col in activity_per_time_range.columns]
    activity_per_time_range = (activity_per_time_range-activity_per_time_range.stack().min())/(activity_per_time_range.stack().max()-activity_per_time_range.stack().min())*2-1
    activity_per_time_range = activity_per_time_range.fillna(0)
    return activity_per_time_range  # .round().astype(int)
def get_device_domain_pca(user_activity_timeseries, n_components=100):
    # Calculate total entries per domain for each device
    domain_entries = user_activity_timeseries.reset_index()[['Device_ID', 'Domain_Name','Activity']].groupby(['Device_ID', 'Domain_Name'])['Activity'].sum()
    domain_entries_pivot = domain_entries.unstack(fill_value=0)
    # domain_entries_pivot = domain_entries_pivot/domain_entries_pivot.sum(axis=1)
    # Normalize the data
    # normalized_data = (domain_entries_pivot - domain_entries_pivot.min().min()) / (domain_entries_pivot.max().max() - domain_entries_pivot.min().min())
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(domain_entries_pivot.fillna(0))
    # Print explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance ratio: {explained_variance.sum():.3f}")
    # Create DataFrame with PCA results
    pca_df = mpd.DataFrame(
        pca_result,
        index=domain_entries_pivot.index,
        columns=[f'pca_domain_{i}' for i in range(n_components)]
    )
    
    return pca_df
#activity fft
from scipy import fft
import numpy as np
from sklearn.preprocessing import StandardScaler
def get_ps_df(user_activity_timeseries,train_devices):
    total_user_activity = user_activity_timeseries.groupby(["Device_ID","Datetime"]).sum()
    total_user_activity = total_user_activity.unstack()
    total_user_activity = total_user_activity.fillna(0)
    power_spectrums = np.abs(fft.fft(total_user_activity.T.values))**2
    sample_d = 3*60*60
    
    freqs = fft.fftfreq(total_user_activity.shape[1],d=sample_d)
    freq_mask = freqs>=0
    power_spectrums = power_spectrums[freq_mask]
    freqs = freqs[freq_mask]

    psd_df = mpd.DataFrame(power_spectrums.T,index=total_user_activity.index,columns=freqs)
    # Get power spectra for training devices only
    train_psd = psd_df.loc[train_devices]

    # Create and fit scaler on training data
    scaler = StandardScaler()
    # scaler.fit(train_psd.T)

    # Transform all data using fitted scaler
    psd_df.iloc[:,:] = scaler.fit_transform(psd_df.values)
    
    return psd_df
# Convert to dataframe and normalize
# power_spectrums_df = power_spectrums.to_frame('power_spectrum')
# power_spectrums_df = (power_spectrums_df - power_spectrums_df.min()) / (power_spectrums_df.max() - power_spectrums_df.min())
