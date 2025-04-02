import modin.pandas as mpd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
def z_normalize_by_all(df,train_devices,per_column = True,fillval=0,fill_na_pre_transform=True, scaler=None):
    if scaler is not None:
        df.iloc[:, :] = scaler.transform(
            df if per_column else df.values.reshape(-1, 1)).reshape(df.shape)
        if fillval:
            df.fillna(fillval,inplace=True)
        return
    scaler = StandardScaler()
    train_data = df.loc[train_devices]
    scaler.fit(train_data if per_column else train_data.values.reshape(-1, 1))

    # Transform all data using fitted scaler
    if fill_na_pre_transform:
        df.fillna(fillval,inplace=True)
    df.iloc[:, :] = scaler.transform(
        df if per_column else df.values.reshape(-1, 1)).reshape(df.shape)
    if fillval is not None:
        df.fillna(fillval,inplace=True)
    params = {
        "mean_": float(scaler.mean_[0]),  # Convert to native Python float
        "var_": float(scaler.var_[0]),
        "scale_": float(scaler.scale_[0]),
        "n_samples_seen_": int(scaler.n_samples_seen_),
    }
    return params
def min_max_scale_all_values(df, train_devices, per_column=False,scaler=None):
    if scaler is not None:
        df.iloc[:, :] = scaler.transform(
            df if per_column else df.values.reshape(-1, 1)).reshape(df.shape)
        return
    # Create MinMaxScaler and fit on training data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data = df.loc[train_devices]
    scaler.fit(train_data if per_column else train_data.values.reshape(-1, 1))

    # Transform all data using fitted scaler
    df.iloc[:, :] = scaler.transform(
        df if per_column else df.values.reshape(-1, 1)).reshape(df.shape)

    # Save scaler parameters as JSON
    scaler_params = {
        "min_": float(scaler.min_[0]),  # Convert to native Python float
        "scale_": float(scaler.scale_[0]),
        "data_min_": float(scaler.data_min_[0]),
        "data_max_": float(scaler.data_max_[0]),
        "data_range_": float(scaler.data_range_[0]),
        # Convert tuple to list for JSON
        "feature_range": list(scaler.feature_range)
    }
    return scaler_params

from sklearn.impute import KNNImputer# , IterativeImputer
from sklearn.linear_model import BayesianRidge
from scipy.sparse import csr_matrix
# from implicit.cpu.als import AlternatingLeastSquares
def impute_missing_values(final_scores_pivot,train_device_ids):
    # Use SoftImpute to fill missing values
    imputer = KNNImputer(n_neighbors=10)
    imputer.fit(final_scores_pivot.loc[train_device_ids])
    imputed_scores = imputer.transform(final_scores_pivot)
    # imputer = AlternatingLeastSquares(factors=10, regularization=0.01, iterations=10,random_state=0)
    # # imputer = MissForest(max_depth=6,max_features=0.8, random_state=0)
    # imputer.fit( user_items = csr_matrix(final_scores_pivot.loc[train_device_ids].values))
    
    # imputed_scores = imputer.recommend_all(csr_matrix(final_scores_pivot.values))
    return imputed_scores
import gc
import ctypes
import sys

def cleanup_memory():
    """
    Force cleanup of memory by:
    1. Running garbage collection
    2. Attempting to release memory back to OS
    """
    # Force garbage collection
    gc.collect()
    
    # Attempt to release memory back to the OS
    if sys.platform.startswith('linux'):
        libc = ctypes.CDLL('libc.so.6')
        # MALLOC_TRIM(0) releases memory back to OS if possible
        print(libc.malloc_trim(0))
