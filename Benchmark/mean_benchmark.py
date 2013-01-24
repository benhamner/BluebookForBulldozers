import numpy as np
import pandas as pd
import util

train, test = util.get_train_test_df()
mean_price = np.mean(train["SalePrice"])
print("The mean price is %0.2f" % mean_price)

util.write_submission("mean_benchmark.csv", [mean_price for i in range(len(test))])