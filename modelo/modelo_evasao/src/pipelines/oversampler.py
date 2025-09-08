from imblearn.over_sampling import SMOTENC

def get_cat_idx(df_x):
    total_cols = df_x.columns.tolist()
    cat_cols = df_x.select_dtypes(include='object').columns.tolist()
    idx_list = []

    for cat_col in cat_cols:
        if cat_col in total_cols:
            idx_list.append(total_cols.index(cat_col))
    
    return idx_list

def get_balanced_data(df_x, df_y, k_neighbors=2):
    smt = SMOTENC(categorical_features=get_cat_idx(df_x), k_neighbors=k_neighbors)
    x_res, y_res = smt.fit_resample(X=df_x, y=df_y)
    return x_res, y_res