import lightgbm as lgb

def train_lgb(X_train, y_train):
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=len(set(y_train)),
        learning_rate=0.1,
        n_estimators=200
    )
    model.fit(X_train, y_train)
    return model