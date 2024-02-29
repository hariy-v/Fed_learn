import pickle
import pandas as pd
import os
import argparse

def main(args):
    folder = args.folder_path
    qty = 500
    cols = os.listdir(folder)
    if "Readme.txt" in cols:
        cols.remove("Readme.txt")
    df = pd.DataFrame()
    for col in cols:
        with open(f"{folder}/{col}/server_side_loss.pkl", "rb") as input_file:
            e = pickle.load(input_file)
        df[col] = e[:qty]
    df["Rounds"] = [i for i in range(len(df))]
    res = df.plot(x="Rounds", y=cols, title="Centralized Test loss").get_figure()
    res.savefig(f'{folder}/Test_loss.png')

    df = pd.DataFrame()
    for col in cols:
        with open(f"{folder}/{col}/server_side_accuracy.pkl", "rb") as input_file:
            e = pickle.load(input_file)
        df[col] = e[:qty]
    df["Rounds"] = [i for i in range(len(df))]
    res = df.plot(x="Rounds", y=cols, title="Centralized Test Accuracy").get_figure()
    res.savefig(f'{folder}/Test_accuracy.png')


    df = pd.DataFrame()
    for col in cols:
        with open(f"{folder}/{col}/server_side_precision.pkl", "rb") as input_file:
            e = pickle.load(input_file)
        df[col] = e[:qty]
    df["Rounds"] = [i for i in range(len(df))]
    res = df.plot(x="Rounds", y=cols, title="Centralized Test Precision").get_figure()
    res.savefig(f'{folder}/Test_precision.png')

    df = pd.DataFrame()
    for col in cols:
        with open(f"{folder}/{col}/server_side_recall.pkl", "rb") as input_file:
            e = pickle.load(input_file)
        df[col] = e[:qty]
    df["Rounds"] = [i for i in range(len(df))]
    res = df.plot(x="Rounds", y=cols, title="Centralized Test Recall").get_figure()
    res.savefig(f'{folder}/Test_recall.png')

    df = pd.DataFrame()
    for col in cols:
        with open(f"{folder}/{col}/server_side_fscore.pkl", "rb") as input_file:
            e = pickle.load(input_file)
        df[col] = e[:qty]
    df["Rounds"] = [i for i in range(len(df))]
    res = df.plot(x="Rounds", y=cols, title="Centralized Test F1score").get_figure()
    res.savefig(f'{folder}/Test_fscore.png')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder-path",
        required=True,
        default=None,
        type=str,
        help="The path for the results",
    )
    args = parser.parse_args()
    main(args)