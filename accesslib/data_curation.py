import pandas as pd
import numpy as np


def json_to_csv(url, csv_columns=None):
    df = pd.read_json(url)
    if csv_columns:
        print(f"Diff columns are: {df.columns.difference(csv_columns)}")
    new_df = df.iloc[:, 1:]
    return new_df


def get_time_code(df_in, col_to_merge, new_col_name):
    df_in[new_col_name] = pd.to_datetime(df_in[col_to_merge].apply(lambda x: "-".join(x.astype(str)), axis=1)).astype(
        'int64') // 10 ** 9
    return df_in


def reindex_df(df_in, column_name_order):
    return df_in.reindex(columns=column_name_order)


def sort_values(df_in, sort_by):
    return df_in.sort_values(by=sort_by)


def equal_dfs_by_code(df_1, df_2):
    """
    Given two dataframes, df_1 with more columns than df_2, returns df_2 with the same columns as df_1, trying to mantain the logic.
    """
    activity_code, sector_code = {}, {}
    for index, row in df_1.iterrows():
        activity_code[row['EPRTRAnnexIMainActivityLabel']] = row['EPRTRAnnexIMainActivityCode']
        sector_code[row['eprtrSectorName']] = row['EPRTRSectorCode']

    df = df_2.copy()
    df['EPRTRAnnexIMainActivityCode'] = 'NaN'
    df['EPRTRSectorCode'] = 'NaN'

    for key in activity_code.keys():
        df['EPRTRAnnexIMainActivityCode'] = np.where(df['EPRTRAnnexIMainActivityLabel'] == key, activity_code[key],
                                                     df['EPRTRAnnexIMainActivityCode'])

    for key in sector_code.keys():
        df['EPRTRSectorCode'] = np.where(df['eprtrSectorName'] == key, sector_code[key], df['EPRTRSectorCode'])

    return df


def check_duplicate(df):
    duplicated_df = df.drop_duplicates()
    print(df.shape[0] - duplicated_df.shape[0])


if __name__ == "__main__":
    # Upload data
    df1 = pd.read_csv("/home/titoare/Documents/ds/seh/data/train/train1.csv")
    df2 = pd.read_csv("/home/titoare/Documents/ds/seh/data/train/train2.csv", delimiter=";")
    check_duplicate(df2)
    df3 = json_to_csv("http://schneiderapihack-env.eba-3ais9akk.us-east-2.elasticbeanstalk.com/first")
    df4 = json_to_csv("http://schneiderapihack-env.eba-3ais9akk.us-east-2.elasticbeanstalk.com/second")
    df5 = json_to_csv("http://schneiderapihack-env.eba-3ais9akk.us-east-2.elasticbeanstalk.com/third")

    # Concatenate data
    df_csv = pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)
    df_json = pd.concat([df3, df4, df5], ignore_index=True)
    df_csv = equal_dfs_by_code(df_json, df_csv)
    df = pd.concat([df_csv, df_json], ignore_index=True)
    check_duplicate(df)

    # Timestamp
    df = get_time_code(df, ['reportingYear', 'MONTH', 'DAY'], "DateTime")

    # Categorical categorical variables to code
    df['cityId_code'] = df["CITY ID"].astype('category').cat.codes
    df['FacilityInspireID_code'] = df["FacilityInspireID"].astype('category').cat.codes
    df['countryName_code'] = df["countryName"].astype('category').cat.codes
    df['City_code'] = df["City"].astype('category').cat.codes
    df['EPRTRAnnexIMainActivityCode_code'] = df["EPRTRAnnexIMainActivityCode"].astype('category').cat.codes
    df['pollutant_code'] = df["pollutant"].astype('category').cat.codes

    # Reindex and sort
    columns_names = ['CITY ID', "cityId_code", 'FacilityInspireID', "FacilityInspireID_code", 'DateTime',
                     'countryName', "countryName_code", 'CONTINENT', 'City', 'City_code',
                     'EPRTRAnnexIMainActivityCode', "EPRTRAnnexIMainActivityCode_code", 'EPRTRAnnexIMainActivityLabel',
                     'EPRTRSectorCode', 'eprtrSectorName',
                     'avg_temp', 'avg_wind_speed', 'max_temp', 'max_wind_speed', 'min_temp', 'min_wind_speed',
                     'targetRelease', 'facilityName',
                     'reportingYear', 'MONTH', 'DAY',
                     'REPORTER NAME', 'DAY WITH FOGS', 'pollutant', "pollutant_code"]

    df = reindex_df(df, columns_names)
    df = sort_values(df, ['CITY ID', 'FacilityInspireID', 'DateTime'])

    # Drop irrelevant columns
    drop_columns = ['CONTINENT', 'EPRTRAnnexIMainActivityLabel', 'EPRTRSectorCode', 'targetRelease',
                    "facilityName", "REPORTER NAME", "DAY WITH FOGS"]
    df.drop(drop_columns, axis=1, inplace=True)

    # Drop two index with NaN string
    df = df.drop(df[df.EPRTRAnnexIMainActivityCode == "NaN"].index)
