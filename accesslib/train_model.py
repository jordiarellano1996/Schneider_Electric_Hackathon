import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

if __name__ == "__main__":
    df = pd.read_csv("/home/titoare/Documents/ds/seh/final4_df.csv", index_col=0)
    df.reset_index(inplace=True, drop=True)

    # Normalize some data
    parameters_to_normalize = ['avg_temp', 'avg_wind_speed']
    for parameter in parameters_to_normalize:
        data = df[parameter].values.reshape(len(df[parameter]), 1)
        df[f"norm_{parameter}"] = MinMaxScaler().fit_transform(data)[:, 0]

    parameters = ['DateTime', 'countryName_code',
                  'EPRTRAnnexIMainActivityCode_code',
                  'eprtrSectorName_code', 'norm_avg_temp', 'norm_avg_wind_speed']
    target = ["pollutant_code"]

    # Split the model, cross_validation, train.
    # I determined 5 folds to have a relation of 80% training, 20% validation.
    # ccp_alphas = [0.00014, 0.00016, 0.00018]
    kf = KFold(n_splits=5, shuffle=False)
    model_list = []

    for train_index, test_index in kf.split(df.values):
        train_x = df.iloc[train_index]
        train_y = df.iloc[train_index]
        validation_x = df.iloc[test_index]
        validation_y = df.iloc[test_index]
        # validation = validation.reshape(len(validation))
        clf = LGBMClassifier()
        model = clf.fit(train_x[parameters].values, train_y[target].values.ravel(),
                        eval_set=[(validation_x[parameters].values, validation_y[target].values.ravel()),
                                  (train_x[parameters], train_y[target].values.ravel())],
                        eval_metric="logloss",
                        )

        predicted_values = model.predict(train_x[parameters].values)
        real_values = train_y[target].values
        f1_result = f1_score(predicted_values, real_values, average="macro")
        print(f"Train f1 score: {f1_result}")

        predicted_values = model.predict(validation_x[parameters].values)
        real_values = validation_y[target].values
        f1_result = f1_score(predicted_values, real_values, average="macro")
        print(f"Validation f1 score: {f1_result}")

        model_list.append(model)
