def weighted_mlp_model():
    from sklearn.model_selection import train_test_split
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.models import Sequential
    from keras import layers                                                                   
    from keras.metrics import AUC, Precision, Recall, BinaryAccuracy
    from load import load_data
    from sklearn.metrics import classification_report
    from keras.losses import BinaryCrossentropy
    from sklearn.utils import compute_class_weight
    import numpy as np

    models = []
    reports = []
    y_tests = []
    y_pred_bins = []
    lbs = ['valence', 'arousal', 'dominance', 'liking']

    for i in range(4):
        features, labels = load_data()
        labels = labels[:, i] 

        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.15, random_state=42)

        mlp = Sequential([
            layers.Input(shape=(32,11)),
            layers.Flatten(), 

            layers.Dense(64, activation=None),
            layers.BatchNormalization(),
            layers.Activation("relu"), 
            layers.Dropout(0.3), 

            layers.Dense(128, activation=None),
            layers.BatchNormalization(),
            layers.Activation("relu"), 
            layers.Dropout(0.4), 

            layers.Dense(1, activation=None)
        ])

        weights = compute_class_weight("balanced", classes=np.arange(2), y=y_train)
        neg_w, pos_w = weights
        sample_weight = np.where(y_train == 1, pos_w, neg_w)

        early_stopping = EarlyStopping(monitor='val_loss', 
                                    patience=10, 
                                    min_delta=0.01, 
                                    verbose=2, 
                                    restore_best_weights=True)
        reduce_plateau = ReduceLROnPlateau(monitor="val_loss", 
                                        factor=0.5, 
                                        patience=5, 
                                        verbose=2)

        mlp.compile(optimizer='adam', 
                    loss = BinaryCrossentropy(from_logits=True), 
                    metrics=[BinaryAccuracy(name="bin_acc"), AUC(name='auc'), Precision(name='precision'), Recall(name='recall')]) 

        mlp.fit(x_train, y_train, validation_split=0.3, epochs=50, sample_weight=sample_weight, callbacks=[early_stopping, reduce_plateau])

        y_pred = mlp.predict(x_test)
        y_pred_bin = (y_pred > 0.5).astype(int)

        mlp.save(f"all_models/mlp_weighted_{lbs[i]}.keras")
        models.append(f"all_models/mlp_weighted_{lbs[i]}.keras")
        reports.append(classification_report(y_test, y_pred_bin, output_dict=True))
        y_tests.append(y_test)
        flattened_y_pred_bin = [item for sublist in y_pred_bin for item in sublist]
        y_pred_bins.append(flattened_y_pred_bin)

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