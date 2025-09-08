from load_data.loadData import load_df_x_y, drop_ir_cols
from sklearn.model_selection import train_test_split
from load_data.getFeatures import get_cat_cols, get_num_cols
from src.pipelines.LRpipeline import build_pipeline
import joblib as jb
from src.pipelines.oversampler import get_balanced_data

df, x, y = load_df_x_y(path='pnad_clean.csv', target_col='evasao', drop_cols=drop_ir_cols)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42, stratify=y)

x_res, y_res = get_balanced_data(df_x=x_train, df_y=y_train)

cat_cols = get_cat_cols(x)
num_cols = get_num_cols(x)

if __name__ == "__main__":
    model = build_pipeline(num_cols=num_cols, cat_cols=cat_cols)

    model.fit(x_res, y_res)

    jb.dump(model, filename='LRModel.joblib')
    print('Modelo exportado')

