import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, plot_confusion_matrix
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv("/home/titoare/Documents/ds/seh/final_data/final_df.csv", index_col=0)
    df.pollutant_code.replace([2, 1, 0], [0, 2, 1], inplace=True)   # Getting standard code.
    df_test = pd.read_csv("/home/titoare/Documents/ds/seh/final_data/final_test_df.csv", index_col=0)
    df_test_native = pd.read_csv("/home/titoare/Documents/ds/seh/final_data/final_test_df.csv", index_col=0)
    df.reset_index(inplace=True, drop=True)

    # Normalize some data
    # parameters_to_normalize = ['avg_temp', 'avg_wind_speed']
    parameters_to_normalize = ['avg_temp', 'avg_wind_speed', 'max_temp', 'max_wind_speed', 'min_temp', 'min_wind_speed']
    for parameter in parameters_to_normalize:
        data_train = df[parameter].values.reshape(len(df[parameter]), 1)
        df[f"norm_{parameter}"] = MinMaxScaler().fit_transform(data_train)[:, 0]
        data_test = df_test[parameter].values.reshape(len(df_test[parameter]), 1)
        df_test[f"norm_{parameter}"] = MinMaxScaler().fit_transform(data_test)[:, 0]

    parameters = ['DateTime', 'countryName_code', "cityId_code",
                  'EPRTRAnnexIMainActivityCode_code',
                  'eprtrSectorName_code', 'norm_max_temp', 'norm_max_wind_speed', 'norm_min_temp', 'norm_min_wind_speed']
    target = ["pollutant_code"]

    # Split the model, cross_validation to hyper tuning parameters.
    # I determined 5 folds to have a relation of 80% training, 20% validation.
    kf = KFold(n_splits=5, shuffle=False)
    model_list = []

    max_depth_arr = [1, 4, 7, 10, 13]
    i = 0
    for train_index, test_index in kf.split(df.values):
        train_x = df.iloc[train_index]
        train_y = df.iloc[train_index]
        validation_x = df.iloc[test_index]
        validation_y = df.iloc[test_index]
        # validation = validation.reshape(len(validation))
        clf = LGBMClassifier(max_depth=max_depth_arr[i], num_leaves=2**max_depth_arr[i])
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
        i += 1

    # Evaluate model
    train_x, validation_x, train_y, validation_y = train_test_split(df[parameters].values, df[target].values,
                                                                    test_size=0.20, random_state=2025)

    clf = LGBMClassifier()
    model = clf.fit(train_x, train_y.ravel(),
                    eval_set=[(validation_x, validation_y.ravel()), (train_x, train_y.ravel())],
                    eval_metric="logloss",
                    )

    predicted_values = model.predict(train_x)
    real_values = train_y
    f1_result = f1_score(predicted_values, real_values, average="macro")
    print(f"Train f1 score: {f1_result}")

    predicted_values = model.predict(validation_x)
    real_values = validation_y
    f1_result = f1_score(predicted_values, real_values, average="macro")
    print(f"Validation f1 score: {f1_result}")

    plot_confusion_matrix(model, train_x, train_y)
    plt.show()
    plot_confusion_matrix(model, validation_x, validation_y)
    plt.show()

    # Create test submission
    test_predicted_values = model.predict(df_test[parameters].values)
    df_test_native["pollutant"] = test_predicted_values
    submission_df = df_test_native[["test_index", "pollutant"]]
    submission_df.to_csv("../predictions.csv", index=False)
    submission_df.to_json("../predictions.json", orient="table",
                          index=False)
