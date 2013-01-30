from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import util

def get_date_dataframe(date_column):
    return pd.DataFrame({
        "SaleYear": [d.year for d in date_column],
        "SaleMonth": [d.month for d in date_column],
        "SaleDay": [d.day for d in date_column]
        }, index=date_column.index)

train, test = util.get_train_test_df()

columns = set(train.columns)
columns.remove("SalesID")
columns.remove("SalePrice")
columns.remove("saledate")

train_fea = get_date_dataframe(train["saledate"])
test_fea = get_date_dataframe(test["saledate"])

for col in columns:
    if train[col].dtype == np.dtype('object'):
        s = np.unique(train[col].values)
        mapping = pd.Series([x[0] for x in enumerate(s)], index = s)
        train_fea = train_fea.join(train[col].map(mapping))
        test_fea = test_fea.join(test[col].map(mapping))
    else:
        train_fea = train_fea.join(train[col])
        test_fea = test_fea.join(test[col])

rf = RandomForestRegressor(n_estimators=50, n_jobs=1, compute_importances = True)
rf.fit(train_fea, train["SalePrice"])
predictions = rf.predict(test_fea)
imp = sorted(zip(train_fea.columns, rf.feature_importances_), key=lambda tup: tup[1], reverse=True)
for fea in imp:
    print(fea)

util.write_submission("random_forest_benchmark.csv", predictions)
