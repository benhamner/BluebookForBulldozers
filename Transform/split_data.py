from dateutil.parser import parse
import os
import pandas as pd

def split_test_set(df, cutoff_time, output_path):
    out_columns = df.columns.delete(df.columns.get_loc("SalePrice"))
    df_public_leaderboard = df[df["saledate"]<=cutoff_time]
    df_public_leaderboard.index = [x for x in range(0, len(df_public_leaderboard))]
    print("There are %d public leaderboard samples" % len(df_public_leaderboard))
    df_final_evaluation = df[df["saledate"]>cutoff_time]
    df_final_evaluation.index = [x for x in range(0, len(df_final_evaluation))]
    print("There are %d final evaluation samples" % len(df_final_evaluation))

    public_labels = pd.DataFrame({"Usage" : 
        ["PublicTest" for i in range(len(df_public_leaderboard))]})
    private_labels = pd.DataFrame({"Usage" : 
        ["PrivateTest" for i in range(len(df_final_evaluation))]})

    df_public_leaderboard = df_public_leaderboard.join(public_labels)
    df_final_evaluation = df_final_evaluation.join(private_labels)

    df_public_leaderboard[out_columns].to_csv(
        os.path.join(output_path, "PublicLeaderboard.csv"), index=False)
    df_public_leaderboard[["SalesID", "SalePrice", "Usage"]].to_csv(
        os.path.join(output_path, "PublicLeaderboardSolution.csv"), index=False)
    df_final_evaluation[out_columns].to_csv(
        os.path.join(output_path, "FinalEvaluation.csv"), index=False)
    df_final_evaluation[["SalesID", "SalePrice", "Usage"]].to_csv(
        os.path.join(output_path, "FinalEvaluationSolution.csv"), index=False)

def main():
    data_path = os.path.join(os.environ["DataPath"], "FastIron")
    raw_path = os.path.join(data_path, "Raw")
    release_path = os.path.join(data_path, "Release")
    test_path = os.path.join(raw_path, "Auction_Solution.csv")
    train_path = os.path.join(raw_path, "Auction_Training.csv")

    cutoff_time = parse("2012-04-30 23:59:59.000")

    df_train = pd.read_csv(train_path, converters={"saledate": parse})
    print("There are %d training samples" % len(df_train))
    df_test = pd.read_csv(test_path, converters={"saledate": parse})
    print("Thre are %d test samples" % len(df_test))

    df_train = df_train.sort("SalesID")
    df_test= df_test.sort("SalesID")

    split_test_set(df_test, cutoff_time, release_path)
    df_train.to_csv(os.path.join(release_path, "Train.csv"), index=False)

if __name__=="__main__":
    main()