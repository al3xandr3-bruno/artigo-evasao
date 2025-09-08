from sklearn.model_selection import GridSearchCV

#grid de busca de hiperparâmetros do LogisticRegression
lr_param_grid = {

}

#grid de busca de hiperparâmetros do MLPClassifier
mlp_param_grid = {
    'mlp_model__learning_rate_init': [0.01, 0.3, 0.1, 0.001],
    'mlp_model__max_iter': [500, 1000, 2000],
    'mlp_model__hidden_layer_sizes': [(50, 50, 50), (80,), (50, 100, 120), (60, 70, 80)]
}

#grid de busca de hiperparâmetros do RandomForestClassifier
rf_param_grid = {
    'rf_model__n_estimators': [100, 200, 300, 400, 500],
    'rf_model__max_depth': [None, 10, 15, 20],
    'rf_model__criterion': ['gini', 'log_loss'],
    'rf_model__min_samples_split': [2, 3, 4],
    'rf_model__max_features': ['sqrt', 'log2'],
    'rf_model__max_leaf_nodes': [None, 2, 3]
}


def get_best_model(model, model_param_grid, cv=10, x_train=None, y_train=None):
    #se o grid estiver vazio, não busca os melhores hiperparams. Apenas treina e retorna o modelo default do pipeline
    if not model_param_grid or model_param_grid is None:
        model.fit(x_train, y_train)
        return model
    
    print('Encontrando melhor modelo . . .')
    gs = GridSearchCV(model, model_param_grid, scoring='accuracy', cv=cv, verbose=2)
    gs.fit(x_train, y_train)

    print('Busca terminada | Melhor modelo selecionado')

    return gs.best_estimator_

