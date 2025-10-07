import json
import pandas as pd
import random
from sklearn.model_selection import train_test_split

def get_dataframes():
    # load JSON
    with open("breast_cancer_groups.json", "r") as f:
        data = json.load(f)

    # convert back to DataFrame
    df = pd.DataFrame(data["records"])

    # split into two groups
    df_0 = df[df["Class"] == 0]
    df_1 = df[df["Class"] == 1]

    # split function for spliting given dataframe into 80:10:10
    def split_group(group):
        train, temp = train_test_split(group, test_size=0.2)
        val, test = train_test_split(temp, test_size=0.5)
        return train, val, test

    # get class 0 sets, and class 1 sets
    train_0, val_0, test_0 = split_group(df_0)
    train_1, val_1, test_1 = split_group(df_1)

    # concatinate class 0 with class 1 sets and shuffle and drop indexes
    train_df = pd.concat([train_0, train_1]).sample(frac=1).reset_index(drop=True).values.tolist()
    val_df   = pd.concat([val_0, val_1]).sample(frac=1).reset_index(drop=True).values.tolist()
    test_df  = pd.concat([test_0, test_1]).sample(frac=1).reset_index(drop=True).values.tolist()

    # shuffle
    random.shuffle(train_df)
    random.shuffle(val_df)
    random.shuffle(test_df)

    return train_df, val_df, test_df
