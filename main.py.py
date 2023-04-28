import matplotlib.pyplot as plt
import matplotlib.ticker as mtick  # For % formatting on the y-axis labels
import numpy as np
import seaborn as sbn  # better plotting and aesthetics

sbn.set_style("ticks")
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


def save(errors) -> None:
    from pathlib import Path

    arr = np.array(errors)
    if len(arr.shape) > 2 or (len(arr.shape) == 2 and 1 not in arr.shape):
        raise ValueError(
            "Invalid output shape. Output should be an array "
            "that can be unambiguously raveled/squeezed."
        )
    if arr.dtype not in [np.float64, np.float32, np.float16]:
        raise ValueError("Your error rates must be stored as float values.")
    arr = arr.ravel()
    if len(arr) != 20 or (arr[0] >= arr[-1]):
        raise ValueError(
            "There should be 20 error values, with the first value "
            "corresponding to k=1, and the last to k=20."
        )
    if arr[-1] >= 2.0:
        raise ValueError(
            "Final array value too large. You have done something "
            "very wrong (probably relating to standardizing)."
        )
    if arr[-1] < 0.8:
        raise ValueError(
            "You probably have not converted your error rates to percent values."
        )
    outfile = Path(__file__).resolve().parent / "errors.npy"
    np.save(outfile, arr, allow_pickle=False)
    print(f"Error rates succesfully saved to {outfile }")


# NOTE: For individual K-values input k_range as range(k, k+1)
def run_knn_models(k_range, tr_data, tr_labels, te_data, te_labels):
    scores = []
    predictions = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(tr_data, tr_labels)
        prediction = knn.predict(te_data)
        predictions.append(prediction)
        scores.append(1 - metrics.accuracy_score(te_labels, prediction))
    return np.array(scores), np.array(predictions)


# Loads in dataset & returns testing data/labels and training data/labels
# Outputs 4 numpy arrays training/testing_data are 2D with a shape of (# of samples, features), labels are 1D with (# of samples) as their shape
def load_dataset(data_path):
    dataset = []
    labels = []

    for line in open(data_path):
        data = line.strip().split(
            ","
        )  # Eliminate whitespace with strip & split into list at commas
        # Grab desired information from samples (remove ID (index 0) & separate label (index 10))
        n_features = len(data) - 2  # Exclude ID & label (2) from n_features
        label = int(data[10])
        data = data[1 : 1 + n_features]
        # If data has no missing attributes, add to the dataset/labels
        if "?" not in data:
            dataset.append(data)
            labels.append(label)

    # Cast dataset and labels to numpy arrays
    dataset = np.array(dataset, dtype=int)
    labels = np.array(labels)

    div_index = 425  # CHANGE THIS HARD CODED VALUE WHEN REVISING DISTRIBUTION OF TRAINING / TESTING SAMPLES
    training_data = dataset[:div_index][:]
    training_labels = labels[:div_index]
    testing_data = dataset[div_index:][:]
    testing_labels = labels[div_index:]

    return training_data, training_labels, testing_data, testing_labels


# Question 2: Describe your dataset. Provide an output for the AUC of the 10 leading measurements
def evaluate_dataset(data_path):
    feature_map = {
        0: ["Clump Thickness"],
        1: ["Uniformity of Cell Size"],
        2: ["Uniformity of Cell Shape"],
        3: ["Marginal Adhesion"],
        4: ["Single Epithelial Cell Size"],
        5: ["Bare Nuclei"],
        6: ["Bland Chromatin"],
        7: ["Normal Nucleoli"],
        8: ["Mitoses"],
    }

    training_data, training_labels, testing_data, testing_labels = load_dataset(
        data_path
    )
    k_val = range(19, 20)  # K=19

    # training_data[:, [0]] isolating the 0th feature from all samples
    n_features = training_data.shape[1]
    auc_scores = []

    for feature in range(n_features):
        scores, predictions = run_knn_models(
            k_val,
            training_data[:, [feature]],
            training_labels,
            testing_data[:, [feature]],
            testing_labels,
        )
        auc = metrics.roc_auc_score(testing_labels, predictions[0])
        feature_map[feature].append(auc)
        auc_scores.append(auc)

    print("--- Feature Analysis ---")
    print("{:<35} {:<35}".format("NAME", "AUC SCORE"))
    for key, value in feature_map.items():
        name, auc_score = value
        auc_score = round(auc_score, 3)
        print("{:<35} {:<35}".format(name, auc_score))


# Question 3: Using KNN model to predict whether samples of breast tumours are benign or malignant
def fit_and_evaluate_model(data_path):
    training_data, training_labels, testing_data, testing_labels = load_dataset(
        data_path
    )

    # Fit and test KNN model with K=19 (highest K-value with the lowest error rate)
    k_range = range(1, 22)
    scores, predictions = run_knn_models(
        k_range, training_data, training_labels, testing_data, testing_labels
    )

    # Graph the recorded accuracy scores
    plt.xlabel("Value of K for KNN")
    plt.ylabel("Error Rate")
    plt.title("Q3: Error Rate in Predicting Cancerous Samples")
    plt.gca().yaxis.set_major_formatter(
        mtick.PercentFormatter(xmax=1.0)
    )  # make y-axis a percentage
    plt.plot(k_range, scores)
    plt.show()
    plt.clf()


if __name__ == "__main__":
    file_name = "breast-cancer-wisconsin.data"
    # setup / helper function calls here, if using
    evaluate_dataset(
        file_name
    )  # these functions can optionally take arguments (e.g. `Path`s to your data)
    fit_and_evaluate_model(file_name)
