import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


temp = [[ 263,    0,    5,    5,    2,   11,    2,    8,    0,    0,   57,   30,    4,    0,     6,   16],
        [  55, 4182,    0,    0,    0,    0,    0,    0,    1,    0,    0,    0,   14,    0,     0,    0],
        [ 530,    3, 3095,   11,    5,  262,   91,    0,    0,    0,    0,   19,    0,    0,     0,    0],
        [  71,    0,   42, 3798,   99,    5,   40,    0,    0,    0,    0,    0,    0,    0,     0,    1],
        [  26,    1,    0,  107, 3280,    0,   55,    0,    0,    0,    0,    0,    0,    0,     0,    0],
        [ 600,   18,  395,   18,    2, 2283,   65,    0,    0,    0,    7,   29,    0,    0,     0,    0],
        [ 658,   12,  176,   29,   43,   75, 1559,    0,    0,    0,    0,    5,    0,    0,     0,    1],
        [ 233,    0,    0,    0,    0,    0,    0,  296,    6,    0,   49,    8,   14,    0,     0,    1],
        [ 156,   11,    0,    0,    0,    0,    0,    0,  206,    0,    0,    0,  822,    0,     0,    0],
        [  54,    0,    2,    0,    0,    2,    0,    0,    0,    1,    3,   76,    0,    0,     2,    0],
        [ 589,   11,   11,    2,    2,   38,   34,   58,    5,    0,  492,   25,   54,    0,     3,   13],
        [ 505,   55,   44,    4,    0,   48,   13,    2,    3,    4,   25,  854,   38,    0,     1,    3],
        [ 153,   16,    0,    0,    0,    0,    0,    0,  167,    0,    0,    3, 4078,    0,     0,    0],
        [ 119,    1,    0,    1,    5,    4,    2,    3,    2,    0,   17,    1,    4,    0,    14,    3],
        [ 177,    0,    1,    0,    4,   12,   12,    7,    2,    0,   10,    5,    1,    1,    45,    5],
        [ 321,    0,    2,    2,    6,   17,    1,    8,    0,    0,   76,   36,   16,    3,    10,   87],]

temp = np.array(temp)

# Generate true labels and predicted labels
true_labels = []
predicted_labels = []

for i in range(temp.shape[0]):  # Iterate over true classes
    for j in range(temp.shape[1]):  # Iterate over predicted classes
        true_labels.extend([i] * temp[i, j])  # Add true class labels
        predicted_labels.extend([j] * temp[i, j])  # Add predicted class labels

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)


# Calculate precision, recall, and F1-score
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels, predicted_labels, average=None
)

# Display results
for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
    print(f"Class {i}: Precision={p:.4f}, Recall={r:.4f}, F1-score={f:.4f}")