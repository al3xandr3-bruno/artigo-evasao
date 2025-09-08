#carrega o dataframe e separa em x, y e df

import pandas as pd

def load_df_x_y(path=None, target_col=None, drop_cols=None): 
    '''
    ### A função carrega os dados de um csv já tratado de acordo com seu path, separa e retorna o dataframe completo (df), as variáveis independentes (x) e a variável dependente (y) 

    - path: caminho do arquivo csv; e.g. path='dados/meu_arquivo.csv' \n
    - target_col: nome da variável dependente (y); e.g. target_col='churn'\n
    - drop_cols: lista com nome das colunas que devem ser removidas no dataframe retornado; e.g. drop_col=['name', 'age']\n

    obs: todos os nomes de variáveis devem ser escritos exatamente como estão na base de dados bruta
    '''

    try:
        df = pd.read_csv(path)
    except:
        print('Erro na transformação do csv em pandas.DataFrame')

    if drop_cols is not None and target_col not in drop_cols:
        df = df.drop(columns=drop_cols)
    
    df_x = df.drop(columns=target_col)
    df_y = df[target_col]

    return df, df_x, df_y


drop_ir_cols = ['frequenta_escola', 'frequentou_escola', 'curso_frequentado', 'terminou_curso', 'num_serie']
