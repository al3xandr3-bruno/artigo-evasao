#captura os tipos das colunas


def get_num_cols(df_x):
    return df_x.select_dtypes(include='number').columns.tolist()

def get_cat_cols(df_x):
    return df_x.select_dtypes(include='object').columns.tolist()
