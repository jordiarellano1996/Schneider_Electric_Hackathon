import numpy as np
import pandas as pd
from accesslib.plot_factory import *

# print(df.query('eprtrSectorName == "Energy sector"').query("pollutant == 'Nitrogen oxides (NOX)'")["pollutant"].count())
# a = df.groupby(["eprtrSectorName", "EPRTRAnnexIMainActivityCode"])["pollutant"].count()


if __name__ == "__main__":
    df = pd.read_csv("/home/titoare/Documents/ds/seh/final4_df.csv", index_col=0)

    parameters = ['DateTime', 'countryName_code',
                  'EPRTRAnnexIMainActivityCode_code',
                  'eprtrSectorName_code', 'avg_temp', 'avg_wind_speed',
                  'pollutant_code']

    # correlation matrix plot
    correlation_table(df[parameters])

    # Plot max temperature in each country
    # country_names = df.countryName.unique()
    # for country_name in country_names:
    #     country_df = df.query(f"countryName == '{country_name}'")
    #     plot_histogram(country_df, "max_temp", country_name)
