import numpy as np
import os
import pickle
from scipy.signal import welch

def load_data(total_subjects=32, subjects_to_load_per_iter=2):
    """
    IMPORTANT: Use "import numpy as np; features.reshape(np.shape(features)[0], -1)" to convert features from 3d to 2d

    Loads DEAP features and labels, with caching to .npy files for speed.
    Loads features of shape: (trials, channels, channel_features)
    If only one of features.npy or labels.npy is found, both are deleted and recomputed.
    """

    features_file = "features.npy"
    labels_file = "labels.npy"

    # Case 1: Both .npy files exist → load them directly
    if os.path.exists(features_file) and os.path.exists(labels_file):
        print("Found cached features and labels .npy files. Loading from disk...")
        features = np.load(features_file)
        labels = np.load(labels_file)
        return features, labels

    # Case 2: Only one file exists → delete both (safety)
    if os.path.exists(features_file) or os.path.exists(labels_file):
        print("Cache inconsistency detected (only one of features.npy or labels.npy exists). Deleting and recomputing...")
        if os.path.exists(features_file):
            os.remove(features_file)
        if os.path.exists(labels_file):
            os.remove(labels_file)

    # Case 3: Neither file exists, compute everything
    print("No cached files found. Computing features and labels...")

    def calculate_skewness(data):
        mean = np.mean(data, axis=2)
        std = np.std(data, axis=2)

        numerator = np.mean((data - mean[:, :, np.newaxis])**3, axis=2)

        denominator = std**3

        return numerator / denominator

    def calculate_bandpower(data, frequency):
        
        # subject_data shape: (trials, channels, timepoints)
        n_trials, n_channels, n_timepoints = data.shape

        bands = np.array([[0.5, 4], [4, 8], [8, 12], [12, 30], [30, 45]])
        n_bands = bands.shape[0]

        relative_bandpower = np.zeros((n_trials, n_channels, n_bands))

        # Calculate Welch's periodogram for all trials and channels at once
        # This reshapes the data to (trials * channels, timepoints) to be passed to welch
        data_reshaped = data.reshape(-1, n_timepoints)
        
        nperseg = int(frequency * 2)
        if nperseg > n_timepoints:
            nperseg = n_timepoints

        f, Pxx = welch(data_reshaped, frequency, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)

        Pxx = Pxx.reshape(n_trials, n_channels, -1)

        total_power = np.sum(Pxx, axis=2, keepdims=True)
        total_power = total_power.reshape(total_power.shape[0], -1)

        for i, band in enumerate(bands):
            fmin, fmax = band
            idx_band = np.logical_and(f >= fmin, f <= fmax)
            relative_bandpower[:, :, i] = np.sum(Pxx[:, :, idx_band], axis=2) / total_power

        return relative_bandpower

    def calculate_kurtosis(data):
        mean = np.mean(data, axis=2)
        std = np.std(data, axis=2)

        numerator = np.mean((data - mean[:, :, np.newaxis])**4, axis=2)

        denominator = std**4

        return ((numerator / denominator) - 3)
        
    def calculate_hjorth_parameters(data):

        eeg_variance = np.var(data, axis=2)

        first_derivative = np.diff(data, axis=2)
        first_derivative_variance = np.var(first_derivative, axis=2)
        mobility = np.sqrt(first_derivative_variance / eeg_variance)

        second_derivative = np.diff(first_derivative, axis=2)
        second_derivative_variance = np.var(second_derivative, axis=2)
        mobility_of_first_derivative = np.sqrt(second_derivative_variance / first_derivative_variance)
        complexity = mobility_of_first_derivative / mobility

        return eeg_variance, mobility, complexity

    def calculate_fractal_dimension(data, kmax):
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
        fractal_dimension = np.array([
            [higuchi_fd(channel, kmax=kmax) for channel in trial]
            for trial in data
        ])
        return fractal_dimension

    fs = 128
    features = []
    labels = []

    subjects = []
    for i in range(0, total_subjects, subjects_to_load_per_iter):
        chunk = list(range(i + 1, min(i + subjects_to_load_per_iter + 1, total_subjects + 1)))
        subjects.append(chunk)

    for subj_ids in subjects: 

        subject_data = []
        labels_data_chunk = []

        for subject_id in subj_ids:
            if subject_id < 10:
                subject_id_str = f"0{subject_id}"
            else:
                subject_id_str = str(subject_id)
                    
            with open(f'deap-dataset/data_preprocessed_python/s{subject_id_str}.dat', 'rb') as file:
                subject = pickle.load(file, encoding='latin1')
                data_per_iter = subject['data'][:, :32, :]
                label_per_iter = (subject['labels'] >= 5).astype(np.int8)
                # subject shape: (trails (40), channels(32), time points((63 seconds * 128 sampling rate) = 8064))
                # labels shape: (trails (40), 4 labels(Valence, Arousal, Dominance, Liking))
                subject_data.append(data_per_iter)
                labels_data_chunk.append(label_per_iter)

        subject_data = np.array(subject_data, dtype=np.float32).reshape(-1, np.shape(subject_data)[2], np.shape(subject_data)[3])
        labels_data_chunk = np.array(labels_data_chunk, dtype=np.int8).reshape(-1, np.shape(labels_data_chunk)[2])

        variance, mobility, complexity = calculate_hjorth_parameters(data=subject_data)
        variance = variance[..., np.newaxis]
        mobility = mobility[..., np.newaxis]
        complexity = complexity[..., np.newaxis]
        skewness = calculate_skewness(data=subject_data)[..., np.newaxis]
        kurtosis = calculate_kurtosis(data=subject_data)[..., np.newaxis]
        fractal_dimension = calculate_fractal_dimension(data=subject_data, kmax=10)[..., np.newaxis]
        rel_bandpower = calculate_bandpower(data=subject_data, frequency=fs)

        feats = np.concatenate([variance, mobility, complexity, skewness, kurtosis, fractal_dimension, rel_bandpower], axis=-1)

        try:  
            features.append(feats)  
        except MemoryError:
            print(
            f'''Memory Error while trying to add {subjects_to_load_per_iter*40} trials features to features list. 
            Current amount of trials in features is {features.shape[0]}. 
            Reduce the subjects_to_load_per_iter variable from {subjects_to_load_per_iter} to something lower.'''
            )
            break
        # IMPORTANT labels.append has to be after try block to ensure extra labels have not been added
        labels.append(labels_data_chunk)

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    # After computation, save to disk
    np.save(features_file, features)
    np.save(labels_file, labels)
    print("Features and labels saved to .npy cache files.")

    return features, labels