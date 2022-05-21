import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("/home/titoare/Documents/ds/seh/final4_df.csv", index_col=0)

    parameters = ['DateTime', 'countryName_code',
                  'EPRTRAnnexIMainActivityCode_code',
                  'eprtrSectorName_code', 'avg_temp', 'avg_wind_speed',
                  'pollutant_code']

    # Normalize some data
    parameters_to_normalize = ['avg_temp', 'avg_wind_speed']

    # Split the model, cross_validation.

