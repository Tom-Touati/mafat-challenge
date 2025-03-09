def get_train_test_masks(domain_counts, test_size=0.2, random_state=42):
    # Get unique device IDs and their corresponding targets
    device_target_df = domain_counts.groupby('Device_ID')['Target'].first().reset_index()
    
    # Perform stratified split on device IDs
    train_device_ids, test_device_ids = train_test_split(
        device_target_df['Device_ID'],
        test_size=test_size,
        random_state=random_state,
        stratify=device_target_df['Target']
    )
    
    # Create mask for train/test split in domain_counts
    train_mask = domain_counts['Device_ID'].isin(train_device_ids)
    test_mask = domain_counts['Device_ID'].isin(test_device_ids)
    
    # Print statistics
    print(f"Total devices: {len(device_target_df)}")
    print(f"Train devices: {len(train_device_ids)}")
    print(f"Test devices: {len(test_device_ids)}")
    print(f"\nTrain samples: {len(domain_counts[train_mask])}")
    print(f"Test samples: {len(domain_counts[test_mask])}")
    
    # Print class distribution
    print("\nTarget distribution in train set:")
    print(domain_counts[train_mask].groupby('Target').size() / len(domain_counts[train_mask]))
    print("\nTarget distribution in test set:")
    print(domain_counts[test_mask].groupby('Target').size() / len(domain_counts[test_mask]))
    
    return train_mask, test_mask

def get_domain_counts(db_df, pivot=False):
    domain_counts = db_df.groupby(["Device_ID","Domain_Name","Target"]).count()
    domain_counts = domain_counts.reset_index()
    domain_counts = domain_counts[["Device_ID","Domain_Name","Target","Datetime"]]
    domain_counts.rename(columns={"Datetime":"count"}, inplace=True)
    if pivot:
        pivot_matrix = train_domain_counts.pivot(index='source', columns='target', values='count').fillna(0)
        return pivot_matrix
    return domain_counts
def get_device_domain_fractions(domain_counts):
    # Calculate total counts per device
    device_totals = domain_counts.groupby('Device_ID')['count'].sum()
    
    # Calculate fractions by dividing each count by the device total
    domain_fractions = domain_counts.copy()
    domain_fractions['fraction'] = domain_fractions.apply(
        lambda row: row['count'] / device_totals[row['Device_ID']], 
        axis=1
    )
    
    return domain_fractions[['Device_ID', 'Domain_Name', 'Target', 'count', 'fraction']]
def compute_domain_target_correlation(domain_counts):
    # Group by Domain_Name and calculate mean Target and count
    domain_stats = domain_counts.groupby('Domain_Name').agg({
        'Target': 'mean',
        'count': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    domain_stats.columns = ['Domain_Name', 'target_mean', 'count_mean', 'count_std', 'n_devices']
    
    # Calculate correlation coefficient
    # We only include domains that appear in multiple devices for statistical significance
    significant_domains = domain_stats[domain_stats['n_devices'] > 1]
    
    # Calculate correlation and p-value
    correlation = mpd.DataFrame({
        'Domain_Name': significant_domains['Domain_Name'],
        'target_correlation': significant_domains['target_mean'],
        'avg_count': significant_domains['count_mean'],
        'count_std': significant_domains['count_std'],
        'n_devices': significant_domains['n_devices']
    }).sort_values('target_correlation', ascending=False)
    
    return correlation
