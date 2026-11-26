"""Create ONE train/test split that every method will use."""
import pandas as pd

from config import DATA_PROCESSED, SEED


def leave_one_out(df, seed=SEED):
    """Hide one random course per user as the test item."""
    shuffled = df.sample(frac=1, random_state=seed)
    test = shuffled.groupby("user").head(1)
    train = shuffled.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def main():
    df = pd.read_csv(DATA_PROCESSED / "interactions.csv")
    train, test = leave_one_out(df)
    train.to_csv(DATA_PROCESSED / "train.csv", index=False)
    test.to_csv(DATA_PROCESSED / "test.csv", index=False)
    print("train:", len(train), "| test:", len(test),
          "| users:", test["user"].nunique())


if __name__ == "__main__":
    main()