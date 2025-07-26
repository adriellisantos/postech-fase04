from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Classes para pipeline

class MinMax(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_scale=['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']):
        self.features_to_scale = features_to_scale
        self.scaler = MinMaxScaler()
        self.features_present_ = None

    def fit(self, X, y=None):
        self.features_present_ = set(self.features_to_scale).issubset(X.columns)
        if self.features_present_:
            self.scaler.fit(X[self.features_to_scale])
        else:
            missing = set(self.features_to_scale) - set(X.columns)
            print(f'Features faltantes para MinMax: {missing}')
        return self

    def transform(self, X):
        X = X.copy()
        if self.features_present_:
            X_scaled = self.scaler.transform(X[self.features_to_scale])
            X[self.features_to_scale] = pd.DataFrame(X_scaled, columns=self.features_to_scale, index=X.index)
        return X
    
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class OneHotEncodingNames(BaseEstimator, TransformerMixin):
    def __init__(self, OneHotEncoding=['CALC', 'CAEC', 'MTRANS']):
        self.OneHotEncoding = OneHotEncoding
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def fit(self, X, y=None):
        if set(self.OneHotEncoding).issubset(X.columns):
            self.encoder.fit(X[self.OneHotEncoding])
        return self

    def transform(self, X):
        X = X.copy()
        if set(self.OneHotEncoding).issubset(X.columns):
            # Transform categorical
            encoded = self.encoder.transform(X[self.OneHotEncoding])
            encoded_df = pd.DataFrame(encoded,
                                      columns=self.encoder.get_feature_names_out(self.OneHotEncoding),
                                      index=X.index)
            # Remove original categorical
            X.drop(columns=self.OneHotEncoding, inplace=True)
            # Concat encoded with rest
            return pd.concat([X, encoded_df], axis=1)
        else:
            print('Uma ou mais features não estão no DataFrame')
            return X

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

class TargetLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, target_col='Obesity'):
        self.target_col = target_col
        self.encoder = LabelEncoder()

    def fit(self, X, y=None):
        self.encoder.fit(X[self.target_col])
        return self

    def transform(self, X):
        X = X.copy()
        encoded = self.encoder.transform(X[self.target_col])
        return pd.DataFrame(encoded, columns=[self.target_col], index=X.index)