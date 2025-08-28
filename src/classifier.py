import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from fl_fall.dataset import clean_datasets, load_data, read_labels
from fl_fall.task import LATENT_SIZE, load_model
from sklearn.model_selection import train_test_split
import numpy as np
import json
import time

import tensorflow as tf
from keras import layers
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
from configs import configs

BUFFER_SIZE = 60000
BATCH_SIZE = 128
base_path = "/home/s7wu7/project/federated_fall_detection/result"
model_base_path = f"{base_path}/centeralized"
result_path = f"{base_path}/classification"

def plot_data_distribution(train_labels, train_users, dataset_name):
    
    partitions, counts = np.unique(train_users, return_counts=True)
    x_axis_values = partitions.astype('int').astype("str")
    activities = np.unique(train_labels)
    green_shades = [
        "#ADFF2F",  # Green Yellow
        "#7FFF00",  # Chartreuse
        "#00FF00",  # Lime
        "#00FF7F",  # Spring Green
        "#00FA9A",  # Medium Spring Green
        "#90EE90",  # Light Green
        "#98FB98",  # Pale Green
        "#66CDAA",  # Medium Aquamarine
        "#20B2AA",  # Light Sea Green
        "#32CD32",  # Lime Green
        "#3CB371",  # Medium Sea Green
        "#2E8B57",  # Sea Green
        "#8FBC8F",  # Dark Sea Green
        "#9ACD32",  # Yellow Green
        "#6B8E23",  # Olive Drab
        "#8A9A5B",  # Moss Green
        "#4F7942",  # Fern Green
        "#228B22",  # Forest Green
        "#556B2F",  # Dark Olive Green
        "#006400"   # Dark Green
    ]
    fig, ax = plt.subplots(figsize=(20, 5))

    until_now = np.zeros(partitions.shape[0])
    for activity_id, color in zip(activities, green_shades[:activities.shape[0]]):
        if activity_id == 0:
            label_name = "ADL"
        elif activity_id == 1:
            label_name = "Fall"
        temp_users, temp_counts = np.unique(train_users[train_labels == activity_id], return_counts=True)
        new_values = np.zeros(partitions.shape[0])
        for temp_user, temp_count in zip(temp_users, temp_counts):
            new_values[partitions == temp_user] += temp_count
        ax.bar(x_axis_values, new_values, bottom=until_now, label=label_name, color=color)
        until_now += new_values

    # Set the labels and title
    ax.set_xlabel('Training User IDs')
    ax.set_ylabel('Count')
    ax.set_title('Per Partition Labels Distribution')
    ax.legend(title='Labels', bbox_to_anchor=(1.28, 1), loc='upper right', borderaxespad=0.)
    
    plt.tight_layout()

    # Show the plot
    plt.savefig(f"{result_path}/{dataset_name}_training_data_distribution.png")

def build_classifier(latent_size):
    input_layer = layers.Input(shape=(latent_size,))
    dense_1 = layers.Dense(512, activation='relu')(input_layer)
    # dense_1 = layers.Dense(512, activation='relu')(input_layer)
    dropout_1 = layers.Dropout(0.3)(dense_1)
    dense_2 = layers.Dense(256, activation='relu')(dropout_1)
    # dense_2 = layers.Dense(128, activation='relu')(dropout_1)
    dropout_2 = layers.Dropout(0.3)(dense_2)
    dense_3 = layers.Dense(64, activation='relu')(dropout_2)
    dropout_3 = layers.Dropout(0.3)(dense_3)
    output_layer = layers.Dense(2)(dropout_3)
    classifier = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return classifier

def build_classifier_adl(latent_size, classes_count):
    input_layer = layers.Input(shape=(latent_size,))
    dense_1 = layers.Dense(256, activation='relu')(input_layer)
    dropout_1 = layers.Dropout(0.3)(dense_1)
    dense_2 = layers.Dense(64, activation='relu')(dropout_1)
    dropout_2 = layers.Dropout(0.3)(dense_2)
    output_layer = layers.Dense(classes_count)(dropout_2)
    classifier = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return classifier

@tf.function
def classifier_train_step(latent_model, classifier, classifier_optimizer, data_batch, label_batch, mertic):
    with tf.GradientTape() as tape:
        latent_features = latent_model(data_batch, training=False)
        latent_features_flattened = tf.reshape(latent_features, (latent_features.shape[0], -1))
        prediction = classifier(latent_features_flattened, training=True)
        loss = tf.keras.losses.categorical_crossentropy(label_batch, prediction, from_logits=True)
        mertic.update_state(label_batch, tf.math.round(tf.math.softmax(prediction)))
        
    gradients = tape.gradient(loss, classifier.trainable_variables)
    classifier_optimizer.apply_gradients(zip(gradients, classifier.trainable_variables))
    
    return loss

# @tf.function
# def classifier_evaluate_step(latent_model, classifier, data_batch, label_batch, metric):
#     latent_features = latent_model(data_batch, training=False)
#     latent_features_flattened = tf.reshape(latent_features, (latent_features.shape[0], -1))
#     prediction = classifier(latent_features_flattened, training=True)
#     loss = tf.keras.losses.categorical_crossentropy(label_batch, prediction, from_logits=True)
#     metric.update_state(label_batch, tf.math.round(tf.math.softmax(prediction)))
#     pred_y = tf.math.round(tf.math.softmax(prediction))
#     return loss, pred_y

@tf.function
def classifier_evaluate_step(latent_model, classifier, data_batch, label_batch, metric):
    latent_features = latent_model(data_batch, training=False)
    latent_features_flattened = tf.reshape(latent_features, (latent_features.shape[0], -1))
    logits = classifier(latent_features_flattened, training=False)
    probs = tf.nn.softmax(logits)
    preds = tf.argmax(probs, axis=1)
    true_labels = tf.argmax(label_batch, axis=1)
    metric.update_state(true_labels, preds)
    loss = tf.keras.losses.categorical_crossentropy(label_batch, logits, from_logits=True)
    return loss, probs, preds, true_labels


# def evaluate_classifier(latent_model, classifier, classification_test_dataset, epoch):
#     m = tf.keras.metrics.Accuracy()
#     epoch_loss_avg = tf.keras.metrics.Mean()
#     y_pred_list = []
#     y_true_list = []
#     batch_size = None
#     for batch_x, batch_y in classification_test_dataset:
#         if batch_size is None:
#             batch_size = batch_x.shape[0]
#         loss, pred_y = classifier_evaluate_step(latent_model, classifier, batch_x, batch_y, m)
#         epoch_loss_avg.update_state(loss)
#         y_pred_list.append(pred_y)
#         y_true_list.append(batch_y)
#         if batch_x.shape[0] != batch_size:
#                 break

#     y_pred = tf.squeeze(tf.concat(y_pred_list, axis=0))
#     y_true = tf.squeeze(tf.concat(y_true_list, axis=0))

#     confusion_matrix = tf.math.confusion_matrix(tf.argmax(y_true, axis=1), tf.argmax(y_pred, axis=1))
    
#     print(f"Epoch {epoch + 1}: Average Testing loss = {epoch_loss_avg.result().numpy()}")
#     print(f"Epoch {epoch + 1}: Average Testing Accuracy = {m.result().numpy()}")
#     print(f"Epoch {epoch + 1}: Average Testing Confusion Matrix = \n {confusion_matrix.numpy()}")
#     return epoch_loss_avg.result().numpy().item(), m.result().numpy().item(), confusion_matrix.numpy().tolist()

def evaluate_classifier(latent_model, classifier, classification_test_dataset, epoch, confidence_threshold=0.99):
    m = tf.keras.metrics.Accuracy()
    epoch_loss_avg = tf.keras.metrics.Mean()

    all_confs = []
    correct_confs = []
    incorrect_confs = []

    y_true_all = []
    y_pred_all = []

    y_true_low_conf = []
    y_pred_low_conf = []

    for batch_x, batch_y in classification_test_dataset:
        loss, probs, preds, true_labels = classifier_evaluate_step(latent_model, classifier, batch_x, batch_y, m)
        epoch_loss_avg.update_state(loss)

        confs = tf.reduce_max(probs, axis=1)
        correct_mask = tf.equal(preds, true_labels)

        # Store confidence data
        correct_confs.extend(tf.boolean_mask(confs, correct_mask).numpy())
        incorrect_confs.extend(tf.boolean_mask(confs, tf.logical_not(correct_mask)).numpy())
        all_confs.extend(confs.numpy())

        # Store labels for full confusion matrix
        y_true_all.extend(true_labels.numpy())
        y_pred_all.extend(preds.numpy())

        # Filter low-confidence samples
        low_conf_mask = confs < confidence_threshold
        y_true_low_conf.extend(tf.boolean_mask(true_labels, low_conf_mask).numpy())
        y_pred_low_conf.extend(tf.boolean_mask(preds, low_conf_mask).numpy())

    # Compute overall metrics
    avg_conf_all = sum(all_confs) / len(all_confs) if all_confs else 0.0
    avg_conf_correct = sum(correct_confs) / len(correct_confs) if correct_confs else 0.0
    avg_conf_incorrect = sum(incorrect_confs) / len(incorrect_confs) if incorrect_confs else 0.0

    # Compute confusion matrices
    cm_all = tf.math.confusion_matrix(y_true_all, y_pred_all).numpy()
    cm_low_conf = tf.math.confusion_matrix(y_true_low_conf, y_pred_low_conf).numpy()

    # Logs
    print(f"Epoch {epoch + 1}: Average Testing loss = {epoch_loss_avg.result().numpy():.4f}")
    print(f"Epoch {epoch + 1}: Accuracy = {m.result().numpy():.4f}")
    print(f"Epoch {epoch + 1}: Avg confidence (All) = {avg_conf_all:.4f}")
    print(f"Epoch {epoch + 1}: Avg confidence (Correct) = {avg_conf_correct:.4f}")
    print(f"Epoch {epoch + 1}: Avg confidence (Incorrect) = {avg_conf_incorrect:.4f}")
    print(f"Epoch {epoch + 1}: Confusion Matrix (All samples):\n{cm_all}")
    print(f"Epoch {epoch + 1}: Confusion Matrix (Confidence < {confidence_threshold}):\n{cm_low_conf}")

    # return {
    #     "loss": epoch_loss_avg.result().numpy().item(),
    #     "accuracy": m.result().numpy().item(),
    #     "avg_conf_correct": avg_conf_correct,
    #     "avg_conf_incorrect": avg_conf_incorrect,
    #     "conf_matrix_all": cm_all.tolist(),
    #     "conf_matrix_low_conf": cm_low_conf.tolist()
    # }
    return epoch_loss_avg.result().numpy().item(), m.result().numpy().item(), cm_all.tolist(), [all_confs, y_pred_all, y_true_all]


def classification(latent_model, classifier, classifier_optimizer, classification_train_dataset, classification_test_dataset, epochs, dataset_name):
    result = {"centralized_evaluate": []}
    confs = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss_avg = tf.keras.metrics.Mean()
        mertic = tf.keras.metrics.Accuracy()   
        batch_size = None 
        for data_batch, label_batch in classification_train_dataset:
            if batch_size is None:
                batch_size = data_batch.shape[0]
            loss = classifier_train_step(latent_model, classifier, classifier_optimizer, data_batch, label_batch, mertic)
            epoch_loss_avg.update_state(loss)
            if data_batch.shape[0] != batch_size:
                break
        # Output the model's average loss for this epoch
        epoch_loss_avg = epoch_loss_avg.result().numpy().item()
        training_metric = mertic.result().numpy().item()
        print(f"Epoch {epoch + 1}: Average Training loss = {epoch_loss_avg}")
        print(f"Epoch {epoch + 1}: Average Training accuracy = {training_metric}")
        test_loss, test_metric, confusion_matrix, all_conf = evaluate_classifier(latent_model, classifier, classification_test_dataset, epoch)
        confs.append(all_conf)
        result['centralized_evaluate'].append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss_avg,
            "train_accuracy": training_metric,
            "test_loss": test_loss,
            "test_accuracy": test_metric,
            "test_confution_matrix": confusion_matrix,
        })

    np.save(f"{result_path}/{dataset_name}_confs.npy", np.array(confs))
    with open(f"{result_path}/{dataset_name}_results.json", "w", encoding="utf-8") as fp:
        json.dump(result, fp)

def main():
    # clean_datasets(True)
    config_name = "SiSFall" # or "UpFall" or "SiSFall"
    # ADL_label_index = 15
    dataset_name = config_name
    user_split = configs[config_name]['user_split']
    frequancy = configs[config_name]['frequancy']
    two_class = configs[config_name]['two_class_classification']
    normlize = configs[config_name]['normlize']
    window_size = configs[config_name]['window_size']
    window_step = configs[config_name]['window_step']
    extract_fall = configs[config_name]['extract_fall']
    epochs = configs[config_name]['epochs']
    balance = configs[config_name]['balance_classification']

    data = load_data(dataset_name=dataset_name, frequancy=frequancy, two_class=two_class, window_size=window_size, 
                    user_split=user_split, window_step=window_step, normlize=normlize, extract_fall=extract_fall,
                    balance=balance, reload=True)

    if user_split:
        test_dataset, train_dataset, test_labels, train_labels, test_users, train_users = data
        plot_data_distribution(train_labels, train_users, f"centeralized_{dataset_name}")
    else:    
        test_dataset, train_dataset, test_labels, train_labels = data

    label_counts = np.unique(train_labels).shape[0]

    classification_train_dataset = tf.data.Dataset.from_tensor_slices((train_dataset, tf.one_hot(train_labels, label_counts))).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    classification_test_dataset = tf.data.Dataset.from_tensor_slices((test_dataset, tf.one_hot(test_labels, label_counts))).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # checkpoint_path = f"{model_base_path}/UpFall_model_state_loss_0.00053_round_90.weights.h5"
    checkpoint_path = f"{model_base_path}/SiSFall_model_state_loss_0.00043_round_50.weights.h5"
    # checkpoint_path = f"{model_base_path}/MobiAct_model_state_loss_0.00359_round_50.weights.h5"
    # checkpoint_path = '/home/s7wu7/project/federated_fall_detection/src/fl-fall/outputs/2025-08-13/10-36-54/model_state_loss_0.00236_round_50.weights.h5'
    model = load_model(window_size)
    model.load_weights(checkpoint_path)
    latent_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('latent_layer').output)
    
    if two_class:
        classifier = build_classifier(latent_size=window_size * LATENT_SIZE)
    else:
        classifier = build_classifier_adl(latent_size=window_size * LATENT_SIZE, classes_count=label_counts)
    print(classifier.summary())
    classifier_optimizer = tf.keras.optimizers.Adam(1e-4)
    classification(latent_model, classifier, classifier_optimizer, classification_train_dataset,
                   classification_test_dataset, epochs, f"centeralized_{dataset_name}")


if __name__ == '__main__':
    main()