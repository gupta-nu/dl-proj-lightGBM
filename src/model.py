import lightgbm as lgb

def train_lgb(X_train, y_train):
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=len(set(y_train)),
        metric='multi_logloss',
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.05,
        min_data_in_leaf=10,
        n_estimators=200,
        verbose=-1
    )
    model.fit(X_train, y_train)
    return model
