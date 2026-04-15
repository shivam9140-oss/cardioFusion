from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

def train_cardio_model(X_train, y_train):
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "Accuracy": accuracy_score(y_test, preds),
        "F1-Score": f1_score(y_test, preds),
        "AUC-ROC": roc_auc_score(y_test, probs),
        "Confusion": confusion_matrix(y_test, preds)
    }
    return metrics
