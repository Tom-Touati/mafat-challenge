import numpy as np
# final_scores_pivot = get_user_domain_scores(domain_activity_timeseries,user_activity_timeseries)


# Test minmax scaler loading
test_data = final_scores_pivot.iloc[0:5, 0:5].copy()

# Save original scaled values
original_scaled = test_data.values.copy()

# Create new scaler and load parameters
new_scaler = MinMaxScaler(feature_range=tuple([-1,1]))  # Use tuple for feature_range
with open('submission/minmax_scaler.json', 'r') as f:
    loaded_params = json.load(f)
    
new_scaler.min_ = np.array([loaded_params["min_"]], dtype=np.float64)
new_scaler.scale_ = np.array([loaded_params["scale_"]], dtype=np.float64)
new_scaler.data_min_ = np.array([loaded_params["data_min_"]], dtype=np.float64)
new_scaler.data_max_ = np.array([loaded_params["data_max_"]], dtype=np.float64)
new_scaler.data_range_ = np.array([loaded_params["data_range_"]], dtype=np.float64)
new_scaler.feature_range = tuple(loaded_params["feature_range"])

# Transform same data with loaded scaler
new_scaled = new_scaler.transform(test_data.values.reshape(-1,1)).reshape(test_data.shape)

# Compare results with higher tolerance
np.testing.assert_array_almost_equal(original_scaled, new_scaled, decimal=6)
print("MinMaxScaler loading test passed!")
