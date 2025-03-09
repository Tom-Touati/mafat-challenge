import numpy as np
import ray
@ray.remote
def calculate_chunk_covariance(matrix_centered, start_idx, end_idx, columns):
    # Calculate covariance for this chunk
    chunk = matrix_centered[:, start_idx:end_idx]
    chunk_cov = np.dot(chunk.T, matrix_centered) / (matrix_centered.shape[0] - 1)
    return start_idx, end_idx, chunk_cov
def compute_chunked_covariance(pivot_matrix, batch_size=2000):
    # Convert to numpy array for faster computation
    matrix_dense = pivot_matrix.to_numpy()
    matrix_centered = matrix_dense - np.mean(matrix_dense, axis=0)

    # Initialize parameters
    n_cols = pivot_matrix.shape[1]
    futures = []

    # Submit tasks to Ray
    for i in range(0, n_cols, batch_size):
        batch_end = min(i + batch_size, n_cols)
        futures.append(calculate_chunk_covariance.remote(matrix_centered, i, batch_end, pivot_matrix.columns))

    # Collect results and combine
    cov_chunks = []
    for future in ray.get(futures):
        start_idx, end_idx, chunk_cov = future
        chunk_df = mpd.DataFrame(
            chunk_cov,
            index=pivot_matrix.columns[start_idx:end_idx],
            columns=pivot_matrix.columns
        )
        cov_chunks.append(chunk_df)

    # Combine all chunks
    return mpd.concat(cov_chunks)
def melt_covariance_matrix(covariance_matrix):
    # Reset index to make it a column
    melted = covariance_matrix.reset_index()
    
    # Melt the dataframe
    melted = melted.melt(
        id_vars=['index'],
        var_name='target',
        value_name='covariance'
    )
    
    # Rename the 'index' column to 'source'
    melted = melted.rename(columns={'index': 'source'})
    
    # Remove duplicate pairs (e.g., if A->B exists, remove B->A)
    melted = melted[melted['source'] < melted['target']]
    
    # Remove rows where source equals target
    melted = melted[melted['source'] != melted['target']]
    
    return melted.reset_index(drop=True)
def filter_and_transform_covariance(melted_covariance, abs_threshold=0.03, log_threshold=10):
    # Create a copy to avoid modifying the original dataframe
    result = melted_covariance
    # Create corrected_cov column
    result['corrected_cov'] = result['covariance'].copy()
    
    # Apply absolute threshold filter
    result.loc[abs(result['corrected_cov']) < abs_threshold, 'corrected_cov'] = 0
    
    # Apply log transformation for values above log_threshold
    high_vals_mask = abs(result['corrected_cov']) > log_threshold
    result.loc[high_vals_mask, 'corrected_cov'] = result.loc[high_vals_mask, 'corrected_cov'].apply(
        lambda x: np.log2(abs(x)) * np.sign(x)
    )
    
    return result
