def weighted_xgb_chain():
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from load import load_data
    import numpy as np 
    from sklearn.utils import compute_class_weight
    import joblib
    
    models = []
    reports = []
    y_tests = []
    y_pred_bins = []
    lbs = ['valence', 'arousal', 'dominance', 'liking']

    for i in range(4):
        features, labels = load_data()
        features = features.reshape(np.shape(features)[0], -1)
        labels = labels[:, i] 

        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.15, random_state=42)

        xgb = XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        use_label_encoder=False,
        eval_metric='logloss', 
        random_state=42
        )
        
        weights = compute_class_weight("balanced", classes=np.arange(2), y=y_train)
        neg_w, pos_w = weights
        sample_weight = np.where(y_train == 1, pos_w, neg_w)

        xgb.fit(x_train, y_train, sample_weight=sample_weight)

        y_pred = xgb.predict(x_test)
        y_pred_bin = (y_pred > 0.5).astype(int)

        joblib.dump(xgb, f"all_models/xgb_weighted_{lbs[i]}.joblib")
        models.append(f"all_models/xgb_weighted_{lbs[i]}.joblib")
        reports.append(classification_report(y_test, y_pred_bin, output_dict=True))
        y_tests.append(y_test)
        y_pred_bins.append(y_pred_bin)

    if __name__ == '__main__':
        for report, lb in zip(reports, lbs): 
            print(f"\n\n{lb}\n{report}")
    
    dictionary = {
    "models": models,
    "reports": reports,
    "y_test": y_tests,
    "y_pred_bins": y_pred_bins
    }

    return dictionary