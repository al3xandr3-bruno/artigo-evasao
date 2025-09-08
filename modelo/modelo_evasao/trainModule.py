from load_data.loadData import load_df_x_y, drop_ir_cols
from sklearn.model_selection import train_test_split
from load_data.getFeatures import get_cat_cols, get_num_cols
from src.pipelines.oversampler import get_balanced_data
import src.pipelines.LRpipeline as lr_pipeline
from src.tuning.gridSearch import get_best_model, mlp_param_grid, rf_param_grid, lr_param_grid
import joblib as jb


#carrega o dataframe e os conjuntos x e y
df, x, y = load_df_x_y(path='pnad_clean.csv', target_col='evasao', drop_cols=drop_ir_cols)

#separa x e y em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42, stratify=y)

#captura as colunas numéricas e categóricas para transformação no pipeline
num_cols = get_num_cols(x)
cat_cols = get_cat_cols(x)

#faz balanceamento de classes (oversampling). APENAS no conjunto de treino
x_res, y_res = get_balanced_data(x_train, y_train)

#função de modelagem de treinamento
def train_and_export(
        model_pipeline=None, 
        num_cols=None, 
        cat_cols=None, 
        model_grid=None, 
        cv_k=10, 
        x_train=None, 
        y_train=None, 
        model_filename=None
        ):
    pre_model = model_pipeline.build_pipeline(num_cols=num_cols, cat_cols=cat_cols)
    model = get_best_model(pre_model, model_param_grid=model_grid, cv=cv_k, x_train=x_train, y_train=y_train)
    if model_filename is None:
        raise NameError('model_filename não pode ser vazio')
    try:
        jb.dump(model, filename=model_filename)
        print(f'Modelo exportado como {model_filename}')
    except:
        print('Exportação falhou. Treine novamente o modelo')

#executa o treinamento (Apenas se este arquivo for diretamente executado)
if __name__ == "__main__":
    train_and_export(
        model_pipeline=lr_pipeline,
        num_cols=num_cols,
        cat_cols=cat_cols,
        model_grid=lr_param_grid,
        x_train=x_res,
        y_train=y_res,
        model_filename='LRModelT.joblib'
    )
