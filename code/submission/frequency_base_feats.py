import numpy as np
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
    active_days = (active_days - active_days.min()) / (
        active_days.max() - active_days.min()) * 2 - 1
    return active_days


def get_activity_per_time_bin(df, bin_hours=3):
    # Convert datetime to time only
    # time_index = db_df.index.to_series().dt.time
    # df["time"] = time_index
    df_copy = df.copy()
    df_copy["time"] = db_df.index.to_series().dt.hour.astype(int) // bin_hours
    df_copy["day_part_activity"] = 0
    activity_per_time_range = df_copy[[
        "Device_ID", "time", "day_part_activity"
    ]].groupby(["Device_ID", "time"]).count()
    activity_per_time_range[
        "activity_fraction"] = activity_per_time_range.groupby(
            "Device_ID").apply(lambda x: x / x.sum()).values
    activity_per_time_range = activity_per_time_range[["activity_fraction"
                                                       ]].reset_index()
    activity_per_time_range = activity_per_time_range.pivot(
        index="Device_ID", columns="time", values="activity_fraction")
    activity_per_time_range.columns = [
        f"time_{col}" for col in activity_per_time_range.columns
    ]
    activity_per_time_range = (
        activity_per_time_range - activity_per_time_range.stack().min()) / (
            activity_per_time_range.stack().max() -
            activity_per_time_range.stack().min()) * 2 - 1
    activity_per_time_range = activity_per_time_range.fillna(0)
    return activity_per_time_range  # .round().astype(int)


#activity fft
from scipy import fft
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_ps_df(db_df, pd,cols=None):
    device_activity_ts = db_df.groupby("Device_ID")["Domain_Name"].resample(
        "3H").count()
    device_activity_ts = device_activity_ts.unstack().fillna(0)
    if cols is not None:
        device_activity_ts = device_activity_ts.reindex(cols,axis="columns").fillna(0)
    # Apply Hann window to the data before FFT
    window = np.hanning(device_activity_ts.shape[1])
    device_activity_ts = device_activity_ts * window
    # device_activity_ts = StandardScaler().fit_transform(device_activity_ts.T)
    power_spectrums = np.abs(fft.rfft(device_activity_ts, axis=1))**2

    sample_d = 3 * 60 * 60

    freqs = fft.rfftfreq(device_activity_ts.shape[1], d=sample_d)
    freq_mask = freqs >= 0
    power_spectrums = power_spectrums[:, freq_mask]
    freqs = freqs[freq_mask]

    psd_df = pd.DataFrame(power_spectrums,
                          index=device_activity_ts.index,
                          columns=freqs)
    # Get power spectra for training devices only

    return psd_df


# Convert to dataframe and normalize
# power_spectrums_df = power_spectrums.to_frame('power_spectrum')
# power_spectrums_df = (power_spectrums_df - power_spectrums_df.min()) / (power_spectrums_df.max() - power_spectrums_df.min())
