def unweighted_random_forest():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from load import load_data
    import joblib
    import numpy as np

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

        rfc = RandomForestClassifier(n_estimators=100, random_state=42) 
        
        rfc.fit(x_train, y_train)

        y_pred = rfc.predict(x_test)
        y_pred_bin = (y_pred > 0.5).astype(int)

        joblib.dump(rfc, f"all_models/rfc_unweighted_{lbs[i]}.joblib")
        models.append(f"all_models/rfc_unweighted_{lbs[i]}.joblib")
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