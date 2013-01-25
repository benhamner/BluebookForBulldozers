import numpy as np
import pandas as pd
import util

train, test = util.get_train_test_df()
median_price = np.median(train["SalePrice"])
print("The median price is %0.2f" % median_price)

util.write_submission("median_benchmark.csv", [median_price for i in range(len(test))])