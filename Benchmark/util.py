from dateutil.parser import parse
import pandas as pd
import os

def get_paths():
    """
    Redefine data_path and submissions_path here to run the benchmarks on your machine
    """
    data_path = os.path.join(os.environ["DataPath"], "FastIron", "Release")
    submission_path = os.path.join(os.environ["DataPath"], "FastIron", "Submissions")
    return data_path, submission_path

def get_train_df(data_path = None):
    if data_path is None:
        data_path, submission_path = get_paths()

    train = pd.read_csv(os.path.join(data_path, "Train.csv"),
        converters={"saledate": parse})
    return train 

def get_test_df(data_path = None):
    if data_path is None:
        data_path, submission_path = get_paths()

    test = pd.read_csv(os.path.join(data_path, "Valid.csv"),
        converters={"saledate": parse})
    return test 

def get_train_test_df(data_path = None):
    return get_train_df(data_path), get_test_df(data_path)

def write_submission(submission_name, predictions, submission_path=None):
    if submission_path is None:
        data_path, submission_path = get_paths()
    
    test = get_test_df()    
    test = test.join(pd.DataFrame({"SalePrice": predictions}))

    test[["SalesID", "SalePrice"]].to_csv(os.path.join(submission_path,
        submission_name), index=False)