from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

def build_pipeline(num_cols, cat_cols):

    col_trans = ColumnTransformer(
        [
            ('cat_trans', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
        ]
    )

    rf_pipeline = Pipeline(
        [
            ('col_trans', col_trans),
            ('rf_model', RandomForestClassifier(
                n_estimators=300,
                n_jobs=-1,
                class_weight='balanced'
                )
            )
        ]
    )

    return rf_pipeline