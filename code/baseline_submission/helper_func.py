import numpy as np
import pandas as pd


def relative_domain(data):
    """
    Feature engeinering: calculate the proportions of all the domain_Name.

    Parameters
    ----------
    data: dataframe
        Browsing data ('Domain_Name' column) of a single Device_ID.

    Returns
    -------
    dataframe
        A dataframe with the proportions for each Domain_Name.
    """
    df = data['Domain_Name'].value_counts(normalize=True)
    df = df.to_frame().T
    df.reset_index(inplace = True, drop = True)
    return df



def cls_proportion(data):
    """
    Feature engeinering: calculate the proportions of all the domain_cls.

    Parameters
    ----------
    data: dataframe
        Browsing data (domain classes' columns) of a single Device_ID.

    Returns
    -------
    dataframe
        A dataframe with the proportions for each Domain_cls.
    """
    combined_cls = data[['Domain_cls1', 'Domain_cls2', 'Domain_cls3', 'Domain_cls4']].values.flatten()
    # Filter out the zeros
    combined_cls = combined_cls[combined_cls != 0]
    df = pd.Series(combined_cls).value_counts(normalize=True)
    df = df.to_frame().T
    df.reset_index(inplace = True, drop = True)
    return df


def avg_relative_entrances_device_id(data, hours_duration):
    """
    Feature engeinering: calculation of the proportional hits according to the day's parts.
    Calculation of proportional hits: sum up the proportional hits for each day's part (calculated each day) and divide them by the number of days (all days of internet usage -queries).

    Parameters
    ----------
    data: dataframe
        Browsing data ('Datetime' column) of a single Device_ID.
    hours_duration : int
        The interval duration of each day's parts in hours (Day division to 24/'hours_duration' parts).

    Returns
    -------
    dataframe
        A dataframe with the proportional hits for each time_range.
    """

    df = pd.to_datetime(data['Datetime'])
    df = df.to_frame()
    df['Datetime'] = df['Datetime'].dt.tz_convert('UTC').dt.tz_localize(None)

    part_length = hours_duration
    num_parts = 24 // part_length

    # Assign part of the day to each timestamp
    df['part_of_day'] = df['Datetime'].dt.hour // part_length

    # Group by date and part of the day, then calculate proportions
    date_groups = df.groupby([df['Datetime'].dt.date, 'part_of_day']).size().unstack(fill_value=0)
    date_groups = date_groups.divide(date_groups.sum(axis=1), axis=0)

    # Add missing parts of the day
    for i in range(num_parts):
        if i not in date_groups.columns:
            date_groups[i] = 0
    date_groups = date_groups.sort_index(axis=1)

    average_proportions = date_groups.sum(axis=0)/date_groups.shape[0]
    return average_proportions.to_frame().T


def corresponding_columns_training_set(df_train_col_list, df):
    """
    This function checks the gaps between the features received as arguments and the data's columns, and changes the columns' data to be the same as those received as arguments.

    Parameters
    ----------
    df_train_col_list: list
        List of features from the training set
    df:  dataframe
        A dataset whose columns will be changed according to df_train_col_list.

    Returns
    -------
    dataframe
        A dataframe with columns compatible with those of the training set.
    """
    del_col = set(list(df.columns)) - set(df_train_col_list)
    df.drop(columns = del_col, inplace = True)
    diff_col = set(df_train_col_list)-set(list(df.columns))
    add_to_test = pd.DataFrame(0, index=np.arange(len(df)), columns=list(diff_col)).astype('float16')
    df = pd.concat([df, add_to_test], axis=1)
    return df


def rename_and_16_convert(dataset, prefix):
    """
    Processing data to reduce memory and creating unique columns names for features.

    Parameters
    ----------
    dataset: dataframe
        A dataframe columns includes Device_ID and features to process.
    prefix : str
        A string to concatenate to the feature names.

    Returns
    -------
    dataframe
        The processed data inclued Device_ID column.
    """
    col_dataset = list(dataset.columns)
    dataset *=1000
    dataset = dataset.astype('float16')

    new_columns_name = {n: f'{prefix}_{n}'for n in col_dataset}

    dataset.rename(columns=new_columns_name, inplace=True)
    return dataset
