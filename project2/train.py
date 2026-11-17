import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression


def card_features(cards):
    card_count = cards.shape[1] // 2
    suits = cards.iloc[:, ::2]
    ranks = cards.iloc[:, 1::2]

    same_suit = 0
    same_rank = 0
    for first in range(card_count):
        for second in range(first + 1, card_count):
            same_suit = same_suit + (suits.iloc[:, first] == suits.iloc[:, second]).astype(int)
            same_rank = same_rank + (ranks.iloc[:, first] == ranks.iloc[:, second]).astype(int)

    return pd.DataFrame({"same_rank": same_rank, "same_suit": same_suit})



def train():
    folder = Path(__file__).resolve().parent
    data = pd.read_csv(folder / "poker-hand-training.csv")
    scores = data.iloc[:, -1]

    for card_count in (3, 4):
        cards = data.iloc[:, :2 * card_count]
        model = LogisticRegression(max_iter=500)
        model.fit(card_features(cards), scores)

        with (folder / f"poker-model{card_count}.sav").open("wb") as file:
            pickle.dump(model, file)
        print(f"Trained the {card_count}-card model on {len(data):,} hands.")


if __name__ == "__main__":
    train()
