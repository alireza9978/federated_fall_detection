import numpy as np
import csv
from sklearn.metrics import confusion_matrix, accuracy_score
# Replace 'your_file.npy' with the path to your .npy file
file_path = '/home/s7wu7/project/federated_fall_detection/result/classification/centeralized_SiSFall_confs.npy'

all_conf = np.load(file_path)

print("Shape of all_conf:", all_conf.shape)
    
for conf_thresh in [0.9, 0.95, 0.96, 0.97, 0.98, 0.99]:
    low_conf = all_conf[-1, :, all_conf[-1, 0] < conf_thresh]
    high_conf = all_conf[-1, :, all_conf[-1, 0] >= conf_thresh]
    print(f"Confidence Threshold: {conf_thresh}")
    print(f"Low Conf samples: {low_conf.shape[0]}")
    print(f"Low Conf Percent: {low_conf.shape[0]/all_conf.shape[-1]}")
    low_conf_pred = low_conf[:, 1]
    low_conf_true = low_conf[:, 2]
    confusion_matrix_low_conf = confusion_matrix(low_conf_true, low_conf_pred)
    print(f"Confusion Matrix for Low Confidence Samples:\n{confusion_matrix_low_conf}")
    high_conf_pred = high_conf[:, 1]
    high_conf_true = high_conf[:, 2]
    confusion_matrix_high_conf = confusion_matrix(high_conf_true, high_conf_pred)
    print(f"Confusion Matrix for High Confidence Samples:\n{confusion_matrix_high_conf}")
    print(f"High Conf samples: {high_conf.shape[0]}")
    print(f"High Conf Percent: {high_conf.shape[0]/all_conf.shape[-1]}")
    print(f"Accuracy for Low Confidence Samples: {accuracy_score(low_conf_true, low_conf_pred)}")
    print(f"Accuracy for High Confidence Samples: {accuracy_score(high_conf_true, high_conf_pred)}")
    total_accuracy = accuracy_score(all_conf[-1, 2, :], all_conf[-1, 1, :])
    print(f"Total Accuracy: {total_accuracy}")
    enhanced_accuracy = accuracy_score(high_conf_true, high_conf_pred) * (high_conf.shape[0]/all_conf.shape[-1]) + \
                        0.91 * (low_conf.shape[0]/all_conf.shape[-1])
    print(f"Enhanced Accuracy: {enhanced_accuracy}")
    print("\n")

    csv_file = '/home/s7wu7/project/federated_fall_detection/result/classification/confidence_analysis.csv'
    header = [
        "Confidence Threshold",
        "Low Conf samples",
        "Low Conf Percent",
        "Low Confusion Matrix",
        "High Conf samples",
        "High Conf Percent",
        "High Confusion Matrix",
        "Accuracy Low Conf",
        "Accuracy High Conf",
        "Total Accuracy",
        "Enhanced Accuracy"
    ]
    rows = []

    for conf_thresh in [0.9, 0.95, 0.96, 0.97, 0.98, 0.99]:
        low_conf = all_conf[-1, :, all_conf[-1, 0] < conf_thresh]
        high_conf = all_conf[-1, :, all_conf[-1, 0] >= conf_thresh]
        low_conf_pred = low_conf[:, 1]
        low_conf_true = low_conf[:, 2]
        confusion_matrix_low_conf = confusion_matrix(low_conf_true, low_conf_pred)
        high_conf_pred = high_conf[:, 1]
        high_conf_true = high_conf[:, 2]
        confusion_matrix_high_conf = confusion_matrix(high_conf_true, high_conf_pred)
        total_accuracy = accuracy_score(all_conf[-1, 2, :], all_conf[-1, 1, :])
        enhanced_accuracy = accuracy_score(high_conf_true, high_conf_pred) * (high_conf.shape[0]/all_conf.shape[-1]) + \
                            0.91 * (low_conf.shape[0]/all_conf.shape[-1])
        row = [
            conf_thresh,
            low_conf.shape[0],
            low_conf.shape[0]/all_conf.shape[-1],
            confusion_matrix_low_conf.tolist(),
            high_conf.shape[0],
            high_conf.shape[0]/all_conf.shape[-1],
            confusion_matrix_high_conf.tolist(),
            accuracy_score(low_conf_true, low_conf_pred),
            accuracy_score(high_conf_true, high_conf_pred),
            total_accuracy,
            enhanced_accuracy
        ]
        rows.append(row)

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)