from load_data.loadData import load_df_x_y, drop_ir_cols
from src.tuning.gridSearch import get_best_model, rf_param_grid
from load_data.getFeatures import get_cat_cols, get_num_cols
from src.pipelines.RFpipeline import build_pipeline
from sklearn.model_selection import train_test_split
import joblib as jb
from src.pipelines.oversampler import get_balanced_data

#carrega os dados de x e y que serão usados paraa treino do modelo
df, x, y = load_df_x_y(path='pnad_clean.csv', target_col='evasao', drop_cols=drop_ir_cols)

num_cols = get_num_cols(x)
cat_cols = get_cat_cols(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

x_res, y_res = get_balanced_data(df_x=x_train, df_y=y_train)

if __name__ == "__main__":

    pre_model = build_pipeline(cat_cols=cat_cols)
    model = get_best_model(model=pre_model, model_param_grid=rf_param_grid, x_train=x_res, y_train=y_res, cv=2)
    jb.dump(model, filename='RFModel.joblib')
