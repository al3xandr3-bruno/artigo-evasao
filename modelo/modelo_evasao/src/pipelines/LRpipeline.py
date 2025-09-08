from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

def build_pipeline(num_cols, cat_cols):
    
    trans_col = ColumnTransformer(
        [
            ('num_trans', StandardScaler(), num_cols),
            ('cat_trans', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
        ]
    )

    lr_pipeline = Pipeline(
        [
            ('trans_col', trans_col),
            ('lr_model', LogisticRegression())
        ]
    )

    return lr_pipeline