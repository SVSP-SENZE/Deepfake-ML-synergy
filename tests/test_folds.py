import pandas as pd

from src.common import FOLDS_CSV
from src.data import train_table


def test_folds_cover_train_exactly():
    folds, df = pd.read_csv(FOLDS_CSV), train_table()
    assert sorted(folds.image_id) == sorted(df.image_id)
    assert folds.image_id.is_unique
    assert set(folds.fold) == {0, 1, 2, 3, 4}


def test_no_cluster_spans_two_folds():
    folds = pd.read_csv(FOLDS_CSV)
    assert (folds.groupby("cluster_id").fold.nunique() == 1).all()


def test_folds_are_balanced():
    folds = pd.read_csv(FOLDS_CSV)
    g = folds.groupby("fold").label.agg(["size", "mean"])
    assert g["size"].between(1300, 1500).all()
    assert g["mean"].between(0.45, 0.56).all()


def test_labels_match_metadata():
    folds, df = pd.read_csv(FOLDS_CSV), train_table()
    m = folds.merge(df, on="image_id", suffixes=("", "_meta"))
    assert (m.label == m.label_meta).all()
