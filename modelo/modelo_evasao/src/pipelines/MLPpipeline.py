#construção de pipelines de dados e do modelo final

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.neural_network import MLPClassifier


def build_pipeline(num_cols=None, cat_cols=None):
    #configura as transformações para variáveis categóricas e numéricas
    col_trans = ColumnTransformer(
        [
            ('num_trans', StandardScaler(), num_cols),
            ('cat_trans', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
        ]
    )

    #execução de sequência e criação do modelo 
    mlp_pipeline = Pipeline(
        [
            ('trans', col_trans),
            ('mlp_model', MLPClassifier(
                hidden_layer_sizes=(30, 20, 20), 
                activation='logistic', 
                learning_rate='constant',
                learning_rate_init=0.01,
                early_stopping=False,
                solver='sgd',
                max_iter=1000
                )
            )
        ]
    )

    return mlp_pipeline