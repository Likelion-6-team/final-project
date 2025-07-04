
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

def make_model(X,Y,cv,file_name):
    
    results = []  # 결과 저장 리스트
    n_splits=cv
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold = 1
    
    Y=pd.Series(Y)
    num_class =Y.nunique()
    
    if num_class==2:
    
        for train_idx, val_idx in cv.split(X, Y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = Y.iloc[train_idx], Y.iloc[val_idx]
        
            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                random_state=fold,
                tree_method="hist",
                device="cuda"
            )
            model.fit(X_tr, y_tr)
        
            # 예측 및 평가
            y_pred = model.predict(X_val)
            report = classification_report(y_val, y_pred, digits=4)
            cm = confusion_matrix(y_val, y_pred)
        
            # 모델 저장
            model.save_model(f"./model/{file_name}_Classifier_{fold}.json")
        
            # 결과 저장
            results.append({
                "fold": fold,
                "confusion_matrix": cm.tolist(),
                "classification_report": report
            })
            
            fold += 1
        
        model.fit(X, Y)
        model.save_model(f"./model/{file_name}_Classifier_final.json")
        

    else:
        for train_idx, val_idx in cv.split(X, Y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = Y.iloc[train_idx], Y.iloc[val_idx]
        
            model = XGBClassifier(
                objective='multi:softmax',
                num_class=num_class,
                eval_metric='mlogloss',
                use_label_encoder=False,
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                tree_method="hist",
                device="cuda")
            model.fit(X_tr, y_tr)
        
            # 예측 및 평가
            y_pred = model.predict(X_val)
            report = classification_report(y_val, y_pred, digits=4)
            cm = confusion_matrix(y_val, y_pred)
        
            # 모델 저장
            model.save_model(f"./model/{file_name}_Classifier_{fold}.json")
        
            # 결과 저장
            results.append({
                "fold": fold,
                "confusion_matrix": cm.tolist(),
                "classification_report": report
            })
        
            fold += 1

        model.fit(X, Y)
        model.save_model(f"./model/{file_name}_Classifier_final.json")
        
    # 결과 JSON 저장 (옵션)
    with open(f"./model/{file_name}_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    return model,results