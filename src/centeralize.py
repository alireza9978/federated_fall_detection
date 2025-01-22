import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from fl_fall.dataset import clean_datasets, load_data, read_labels
from fl_fall.task import load_model
from sklearn.model_selection import train_test_split
import json
import matplotlib
from configs import configs

BUFFER_SIZE = 60000
BATCH_SIZE = 128
result_path = "/home/s7wu7/project/federated_fall_detection/result/centeralized"


def plot_data_distribution(train_labels, train_users, dataset_name):
    # matplotlib.rcParams.update({'font.size': 32})
    
    label_df = read_labels()
    if dataset_name == "UpFall":
        label_df_dataset_name = "UP FALL"
    elif dataset_name == "SiSFall":
        label_df_dataset_name = "SIS FALL"
    
    partitions, counts = np.unique(train_users, return_counts=True)
    x_axis_values = partitions.astype('int').astype("str")
    activities = np.unique(train_labels)
    color_gradient = [
        "#FF4500",  # Orange Red
        "#FF6347",  # Tomato
        "#FF7F50",  # Coral
        "#FFA07A",  # Light Salmon
        "#FFD700",  # Gold
        "#FFFF00",  # Yellow
        "#ADFF2F",  # Green Yellow
        "#7FFF00",  # Chartreuse
        "#00FF00",  # Lime
        "#00FA9A",  # Medium Spring Green
        "#00CED1",  # Dark Turquoise
        "#1E90FF",  # Dodger Blue
        "#4169E1",  # Royal Blue
        "#6A5ACD",  # Slate Blue
        "#9370DB",  # Medium Purple
        "#BA55D3",  # Medium Orchid
        "#D87093",  # Pale Violet Red
        "#FF69B4",  # Hot Pink
        "#FF1493",  # Deep Pink
        "#FF4500",  # Orange Red (loop back to start for a circular gradient)
        "#8B0000",  # Dark Red
        "#8B4513",  # Saddle Brown
        "#A0522D",  # Sienna
        "#CD5C5C",  # Indian Red
        "#F08080",  # Light Coral
        "#FFDAB9",  # Peach Puff
        "#EEE8AA",  # Pale Goldenrod
        "#98FB98",  # Pale Green
        "#AFEEEE",  # Pale Turquoise
        "#DDA0DD"   # Plum
    ]

    fig, ax = plt.subplots(figsize=(40, 25))
    w = 0.3  # Bar width

    until_now = np.zeros(partitions.shape[0])
    for activity_id, color in zip(activities[::-1], color_gradient[:activities.shape[0]]):
        # label_name = label_df[label_df[label_df_dataset_name] == activity_id]['ACTIVITY'].values[0]
        label_name = str(int(activity_id))
        temp_users, temp_counts = np.unique(train_users[train_labels == activity_id], return_counts=True)
        new_values = np.zeros(partitions.shape[0])
        for temp_user, temp_count in zip(temp_users, temp_counts):
            new_values[partitions == temp_user] += temp_count
        ax.bar(x_axis_values, new_values, align='center', width=w, bottom=until_now, label=label_name, color=color)
        until_now += new_values

    # Set the labels and title
    ax.set_xlabel('Training User IDs')
    ax.set_ylabel('Count')
    ax.set_title('Per Partition Labels Distribution')
    ax.legend(title='Labels', bbox_to_anchor=(1.08, 1), loc='upper right', borderaxespad=0.)
    
    # Adjust margins and limits to reduce gaps
    ax.set_xlim(-0.5, len(partitions) - 0.5)
    plt.margins(x=0.01)  # Reduce default margins
    plt.tight_layout()
    plt.grid(True)

    # Show the plot
    plt.savefig(f"{result_path}/{dataset_name}_training_data_distribution.png")

        
def generate_and_save_images(model, epoch, test_input, dataset_name, sample_numbers=10):
    reconstructed = model(test_input, training=False)
    
    original = test_input.numpy()
    reconstructed = reconstructed.numpy()
    
    fig, axs = plt.subplots(sample_numbers, original.shape[2], figsize=(15, sample_numbers * 3))
    
    for sample_idx in range(sample_numbers):
        for feature_idx in range(original.shape[2]):
            axs[sample_idx, feature_idx].plot(original[sample_idx, :, feature_idx], label='Original', alpha=0.6)
            axs[sample_idx, feature_idx].plot(reconstructed[sample_idx, :, feature_idx], label='Reconstructed', linestyle='dashed')
            if sample_idx == 0:
                axs[sample_idx, feature_idx].set_title(f'Feature {feature_idx + 1}')
            axs[sample_idx, feature_idx].legend()

    # plt.tight_layout()
    plt.savefig(f"{result_path}/{dataset_name}_reconstructed_epoch_{epoch}.png")
    plt.close()

@tf.function
def train_step(model, model_optimizer, data_batch):
    with tf.GradientTape() as tape:
        reconstructed = model(data_batch, training=True)
        loss = tf.keras.losses.MeanSquaredError()(data_batch, reconstructed)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    model_optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

def evaluate_model(model, test_dataset, epoch):
    epoch_loss_avg = tf.keras.metrics.Mean()
    batch_size = None
    for batch_x in test_dataset:
        if batch_size is None:
            batch_size = batch_x.shape[0]
        reconstructed = model(batch_x, training=False)

        batch_loss = tf.keras.losses.MeanSquaredError()(batch_x, reconstructed)

        epoch_loss_avg.update_state(batch_loss)

        if batch_x.shape[0] != batch_size:
            break
    
    print(f"Epoch {epoch}: Average Testing loss = {epoch_loss_avg.result().numpy()}")
    return epoch_loss_avg.result().numpy().item()
        
def train(model, model_optimizer, dataset, test_dataset, epochs, dataset_name):
    result = {"centralized_evaluate": []}
    for epoch in range(epochs):
        
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss_avg = tf.keras.metrics.Mean()
        batch_size = None
        for data_batch in dataset:
            if batch_size is None:
                batch_size = data_batch.shape[0]
            loss = train_step(model, model_optimizer, data_batch)
            epoch_loss_avg.update_state(loss)
            if data_batch.shape[0] != batch_size:
                break
        
        epoch_loss_avg = epoch_loss_avg.result().numpy().item()
        print(f"Epoch {epoch + 1}: Average Training loss  = {epoch_loss_avg}")
        
        test_loss = evaluate_model(model, test_dataset, epoch + 1)
        result['centralized_evaluate'].append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss_avg,
            "test_loss": test_loss
        })
        
        if epoch % 10 == 9:
            generate_and_save_images(model, epoch + 1, next(iter(test_dataset.take(1))), dataset_name)
            file_name = f"{result_path}/{dataset_name}_model_state_loss_{epoch_loss_avg:.5f}_round_{epoch + 1}.weights.h5"
            model.save_weights(file_name)
    
    with open(f"{result_path}/{dataset_name}_results.json", "w", encoding="utf-8") as fp:
        json.dump(result, fp)
        
        
def main():
    # clean_datasets(True)
    config_name = "MobiAct" # or "UpFall"
    # ADL_label_index = 15
    dataset_name = config_name
    user_split = configs[config_name]['user_split']
    frequancy = configs[config_name]['frequancy']
    two_class = configs[config_name]['two_class']
    normlize = configs[config_name]['normlize']
    window_size = configs[config_name]['window_size']
    window_step = configs[config_name]['window_step']
    extract_fall = configs[config_name]['extract_fall']
    epochs = configs[config_name]['epochs']
    balance = configs[config_name]['balance']

    data = load_data(dataset_name=dataset_name, frequancy=frequancy, two_class=two_class, window_size=window_size, 
                    user_split=user_split, window_step=window_step, normlize=normlize, extract_fall=extract_fall,
                    balance=balance, reload=False)
    if user_split:
        train_dataset, _, train_labels, _, train_users, _ = data
        # train_dataset = train_dataset[train_labels > ADL_label_index]
        # train_users = train_users[train_labels > ADL_label_index]
        # train_labels = train_labels[train_labels > ADL_label_index]
        plot_data_distribution(train_labels, train_users, dataset_name)
    else:    
        train_dataset, _, train_labels, _ = data
        # train_dataset = train_dataset[train_labels == 0]
        # train_labels = train_labels[train_labels == 0]
    
    print(np.unique(train_labels, return_counts=True))
    
    x_train, x_val = train_test_split(train_dataset, test_size=0.15, random_state=42)
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(x_val).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    
    model = load_model(window_size)
    print(model.summary())
    model_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    train(model, model_optimizer, train_dataset, test_dataset, epochs, dataset_name)
    
    
if __name__ == '__main__':
    main()