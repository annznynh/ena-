import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TabularEncoder(TransformerMixin, BaseEstimator):
    """Frozen training categories and numerical medians; unseen values get zero."""

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        self.categories_ = {}
        self.medians_ = {}
        self.onehot_ = {}
        self.output_names_ = []
        for col in self.columns_:
            if X[col].dtype == object or isinstance(X[col].dtype, pd.StringDtype):
                counts = X[col].fillna("__UNK__").value_counts()
                if col.endswith("skill"):
                    self.categories_[col] = (counts / len(X)).to_dict()
                    self.output_names_.append(col)
                else:
                    self.onehot_[col] = sorted(set(counts.index) | {"__UNK__"})
                    self.output_names_.extend(
                        col + "=" + str(v) for v in self.onehot_[col]
                    )
            else:
                median = X[col].median()
                self.medians_[col] = float(median) if pd.notna(median) else 0.0
                self.output_names_.append(col)
        return self

    def transform(self, X):
        out = np.empty((len(X), len(self.output_names_)), dtype=np.float32)
        i = 0
        for col in self.columns_:
            if col in self.onehot_:
                values = X[col].fillna("__UNK__")
                values = values.where(values.isin(self.onehot_[col]), "__UNK__")
                for category in self.onehot_[col]:
                    out[:, i] = values.eq(category).to_numpy()
                    i += 1
                continue
            if col in self.categories_:
                out[:, i] = (
                    X[col]
                    .fillna("__UNK__")
                    .map(self.categories_[col])
                    .fillna(0)
                    .to_numpy()
                )
            else:
                out[:, i] = X[col].fillna(self.medians_[col]).to_numpy()
            i += 1
        if not np.isfinite(out).all():
            raise ValueError("Nonfinite model features")
        return out
