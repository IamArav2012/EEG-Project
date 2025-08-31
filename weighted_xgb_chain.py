import pickle
import numpy as np 
from xgboost import XGBClassifier
from sklearn.multioutput import ClassifierChain
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

fs = 128
subjects = list(range(1, 7, 1))
subject_data = []
labels_data = []

# Loop through each subject
for subject_id in subjects:
    if subject_id < 10:
        subject_id_str = f"0{subject_id}"
    else:
        subject_id_str = str(subject_id)
        
    with open(f'deap-dataset/data_preprocessed_python/s{subject_id_str}.dat', 'rb') as file:
        subject = pickle.load(file, encoding='latin1')
        data = subject['data'][:, :32, :]
        labels = (subject['labels'] >= 5).astype(np.int8)
        # subject shape: (trails (40), channels(32), time points((63 seconds * 128 sampling rate) = 8064))
        # labels shape: (trails (40), 4 labels(Valence, Arousal, Dominance, Liking))
    
    subject_data.append(data)
    labels_data.append(labels)

subject_data = np.array(subject_data, dtype=np.float32)
labels_data = np.array(labels_data, dtype=np.int8)
subject_data = subject_data.reshape(-1, np.shape(subject_data)[2], np.shape(subject_data)[3])
labels_data = labels_data.reshape(-1, np.shape(labels_data)[2])
subject_data = subject_data[:200,:]
labels_data = labels_data[:200,:]

def calculate_skewness(subject_data):
    mean = np.mean(subject_data, axis=2)
    std = np.std(subject_data, axis=2)

    # Vectorized calculation of the numerator (mean of the cubed deviations)
    numerator = np.mean((subject_data - mean[:, :, np.newaxis])**3, axis=2)

    denominator = std**3

    return numerator / denominator

def calculate_bandpower(subject_data):
    
    # subject_data shape: (trials, channels, timepoints)
    n_trials, n_channels, n_timepoints = subject_data.shape

    # Define the frequency bands
    bands = np.array([[0.5, 4], [4, 8], [8, 12], [12, 30], [30, 45]])
    n_bands = bands.shape[0]

    # Pre-allocate the output array
    relative_bandpower = np.zeros((n_trials, n_channels, n_bands))

    # Calculate Welch's periodogram for all trials and channels at once
    # This reshapes the data to (trials * channels, timepoints) to be passed to welch
    data_reshaped = subject_data.reshape(-1, n_timepoints)
    
    nperseg = int(fs * 2)
    if nperseg > n_timepoints:
        nperseg = n_timepoints

    f, Pxx = welch(data_reshaped, fs, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)

    # Reshape Pxx back to its original trial/channel structure
    Pxx = Pxx.reshape(n_trials, n_channels, -1)

    total_power = np.sum(Pxx, axis=2, keepdims=True)
    total_power = total_power.reshape(total_power.shape[0], -1)

    # Calculate bandpower for all trials, channels, and bands simultaneously
    for i, band in enumerate(bands):
        fmin, fmax = band
        idx_band = np.logical_and(f >= fmin, f <= fmax)
        relative_bandpower[:, :, i] = np.sum(Pxx[:, :, idx_band], axis=2) / total_power

    return relative_bandpower

def calculate_kurtosis(subject_data):
    mean = np.mean(subject_data, axis=2)
    std = np.std(subject_data, axis=2)

    # Vectorized calculation of the numerator
    numerator = np.mean((subject_data - mean[:, :, np.newaxis])**4, axis=2)

    denominator = std**4

    return ((numerator / denominator) - 3)
    
def calculate_hjorth_parameters(subject_data):

    eeg_variance = np.var(subject_data, axis=2)

    first_derivative = np.diff(subject_data, axis=2)
    first_derivative_variance = np.var(first_derivative, axis=2)
    mobility = np.sqrt(first_derivative_variance / eeg_variance)

    second_derivative = np.diff(first_derivative, axis=2)
    second_derivative_variance = np.var(second_derivative, axis=2)
    mobility_of_first_derivative = np.sqrt(second_derivative_variance / first_derivative_variance)
    complexity = mobility_of_first_derivative / mobility

    return eeg_variance, mobility, complexity

def calculate_fractal_dimension(subject_data, kmax):
    def higuchi_fd(x, kmax):
        """
        Compute Higuchi Fractal Dimension of a time series.
        
        Parameters:
            x : array-like
                1D time series
            kmax : int
                Maximum k (scale) to consider
                
        Returns:
            HFD value (float)
        """
        N = len(x)
        Lk = np.zeros(kmax)
        x = np.array(x)

        for k in range(1, kmax+1):
            Lm = []
            for m in range(k):
                idxs = np.arange(1, int(np.floor((N-m)/k)), dtype=np.int32)
                length = np.sum(np.abs(x[m + idxs * k] - x[m + k * (idxs - 1)]))
                length *= (N - 1) / ( (len(idxs) * k) * k )
                Lm.append(length)
            Lk[k-1] = np.mean(Lm)

        lnLk = np.log(Lk)
        lnk = np.log(1.0 / np.arange(1, kmax+1))

        # Linear fit to estimate slope (fractal dimension)
        hfd = np.polyfit(lnk, lnLk, 1)[0]
        return hfd
    fractal_dimension = [
    [higuchi_fd(channel, kmax=kmax) for channel in trial]
    for trial in subject_data
]
    return fractal_dimension

variance, mobility, complexity = calculate_hjorth_parameters(subject_data=subject_data)
skewness = calculate_skewness(subject_data=subject_data)
kurtosis = calculate_kurtosis(subject_data=subject_data)
rel_bandpower = calculate_bandpower(subject_data=subject_data)
fractal_dimension = calculate_fractal_dimension(subject_data=subject_data, kmax=10)

features = np.stack([variance, mobility, complexity, skewness, kurtosis, fractal_dimension], axis=2)
features = np.concatenate([features, rel_bandpower], axis=2)
# The structure is now (variance, mobility, complexity, skewness, kurtosis, fractal_dimension, bandpower_delta, bandpower_theta, bandpower_alpha, bandpower_beta, bandpower_gamma)

# Make compatible with XGB (dim <=2)
features = features.reshape(features.shape[0], -1)

# Split your data (features and labels_data)
x_train, x_test, y_train, y_test = train_test_split(features, labels_data, test_size=0.15, random_state=42)

class_weights = []
for i in range(y_train.shape[1]):  # Iterate over each label in the multi-label setup
    weights = compute_class_weight('balanced', classes=np.arange(2), y=y_train[:, i])
    class_weights.append(weights)

# Define a function to create an XGBClassifier with class weights
def create_xgb_with_class_weights(class_weight):
    return XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=class_weight[1] / class_weight[0],  # class 1 / class 0 weight ratio
        random_state=42
    )

classifiers = [create_xgb_with_class_weights(class_weight) for class_weight in class_weights]

# Define the classifier chain with the base classifiers
chain = ClassifierChain(base_estimator=classifiers[0], order=list(range(4)), random_state=42)

# Instead of setting `estimators_`, we directly use a custom chain with individual classifiers
chain.estimators_ = classifiers

chain.fit(x_train, y_train)

y_pred = chain.predict(x_test)
report = classification_report(y_test, y_pred)
print(report)
'''
precision    recall  f1-score   support

           0       0.67      0.20      0.31        10
           1       1.00      0.29      0.44         7
           2       0.69      0.64      0.67        14
           3       0.72      0.82      0.77        22

   micro avg       0.72      0.58      0.65        53
   macro avg       0.77      0.49      0.55        53
weighted avg       0.74      0.58      0.61        53
 samples avg       0.58      0.50      0.52        53
'''