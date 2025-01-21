import json
import matplotlib
import matplotlib.pyplot as plt


def plot_classification_training_result(file_path):
    base_path = "/home/s7wu7/project/federated_fall_detection/result/classification"
        # Load JSON data from a file
    with open(f"{base_path}/{file_path}", 'r') as file:
        data = json.load(file)
        
    # Extract data
    epochs = [entry["epoch"] for entry in data["centralized_evaluate"]]
    train_loss = [entry["train_loss"] for entry in data["centralized_evaluate"]]
    test_loss = [entry["test_loss"] for entry in data["centralized_evaluate"]]
    train_accuracy = [entry["train_accuracy"] for entry in data["centralized_evaluate"]]
    test_accuracy = [entry["test_accuracy"] for entry in data["centralized_evaluate"]]

    # Calculate Precision, Recall, and F1 Score
    precision_list = []
    recall_list = []
    f1_score_list = []

    for entry in data["centralized_evaluate"]:
        conf_matrix = entry["test_confution_matrix"]
        TN, FP = conf_matrix[0]
        FN, TP = conf_matrix[1]

        # Calculate precision, recall, and F1
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)

    # Plotting Train and Test Loss
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.title('Train and Test Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{base_path}/{file_path[:-13]}_loss.png")

    # Plotting Train and Test Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_accuracy, label='Train Accuracy')
    plt.plot(epochs, test_accuracy, label='Test Accuracy')
    plt.title('Train and Test Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{base_path}/{file_path[:-13]}_accuracy.png")

    # Plotting Precision, Recall, and F1 Score
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, precision_list, label='Precision')
    plt.plot(epochs, recall_list, label='Recall')
    plt.plot(epochs, f1_score_list, label='F1 Score')
    plt.title('Precision, Recall, and F1 Score Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{base_path}/{file_path[:-13]}_metrics.png")

    
def plot_reconstruction_training_result(file_path):
    base_path = "/home/s7wu7/project/federated_fall_detection/result/centeralized"
    # Load JSON data from a file
    with open(f"{base_path}/{file_path}", 'r') as file:
        data = json.load(file)

    # Extract data
    epochs = [entry["epoch"] for entry in data["centralized_evaluate"]]
    train_loss = [entry["train_loss"] for entry in data["centralized_evaluate"]]
    test_loss = [entry["test_loss"] for entry in data["centralized_evaluate"]]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Test Loss')

    # Adding title and labels
    plt.title('Train and Test Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.savefig(f"{base_path}/{file_path[:-5]}.png")
        
        
def plot_test_metrics(federated_file_path, centeralazied_file_path):
    # Plotting Precision, Recall, and F1 Score
    fig = plt.figure(figsize=(18, 24))
    
    plots = fig.subplots(4, 1, sharex=True)

    if federated_file_path is not None:
        base_path = "/home/s7wu7/project/federated_fall_detection/result/classification"
        # Load JSON data from a file
        with open(f"{base_path}/{federated_file_path}", 'r') as file:
            data = json.load(file)
            
        # Extract data
        fl_epochs = [entry["epoch"] for entry in data["centralized_evaluate"]]
        fl_test_accuracy = [entry["test_accuracy"] for entry in data["centralized_evaluate"]]

        # Calculate Precision, Recall, and F1 Score
        fl_precision_list = []
        fl_recall_list = []
        fl_f1_score_list = []

        for entry in data["centralized_evaluate"]:
            conf_matrix = entry["test_confution_matrix"]
            TN, FP = conf_matrix[0]
            FN, TP = conf_matrix[1]

            # Calculate precision, recall, and F1
            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            recall = TP / (TP + FN) if (TP + FN) != 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

            fl_precision_list.append(precision)
            fl_recall_list.append(recall)
            fl_f1_score_list.append(f1_score)
        
        plots[0].plot(fl_epochs, fl_test_accuracy, linewidth=5, label='Federated Accuracy')
        plots[1].plot(fl_epochs, fl_precision_list, linewidth=5, label='Federated Precision')
        plots[2].plot(fl_epochs, fl_recall_list, linewidth=5, label='Federated Recall')
        plots[3].plot(fl_epochs, fl_f1_score_list, linewidth=5, label='Federated F1 Score')

    if centeralazied_file_path is not None:
        base_path = "/home/s7wu7/project/federated_fall_detection/result/classification"
        with open(f"{base_path}/{centeralazied_file_path}", 'r') as file:
            data = json.load(file)    

        # Extract data
        epochs = [entry["epoch"] for entry in data["centralized_evaluate"]]
        test_accuracy = [entry["test_accuracy"] for entry in data["centralized_evaluate"]]

        # Calculate Precision, Recall, and F1 Score
        precision_list = []
        recall_list = []
        f1_score_list = []

        for entry in data["centralized_evaluate"]:
            conf_matrix = entry["test_confution_matrix"]
            TN, FP = conf_matrix[0]
            FN, TP = conf_matrix[1]

            # Calculate precision, recall, and F1
            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            recall = TP / (TP + FN) if (TP + FN) != 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)

        plots[0].plot(epochs, test_accuracy, linewidth=5, label='Accuracy')
        plots[1].plot(epochs, precision_list, linewidth=5, label='Precision')
        plots[2].plot(epochs, recall_list, linewidth=5, label='Recall')
        plots[3].plot(epochs, f1_score_list, linewidth=5, label='F1 Score')

    # plots[0].set_title('Test Accuracy, Precision, Recall, and F1 Score Over Epochs')
    plots[0].legend(frameon=False)
    plots[1].legend(frameon=False)
    plots[2].legend(frameon=False)
    plots[3].legend(frameon=False)
    plots[3].set_xlabel('Epoch')
    plots[0].set_ylabel('Accuracy Score')
    plots[1].set_ylabel('Precision Score')
    plots[2].set_ylabel('Recall Score')
    plots[3].set_ylabel('F1 Score')
    
    plt.tight_layout()
    plt.savefig(f"model_metrics.png", transparent=True)

def plot_test_losses():
    file_path = "/home/s7wu7/project/federated_fall_detection/result/centeralized/SiSFall_results.json"
    # Load JSON data from a file
    with open(file_path, 'r') as file:
        data = json.load(file)
    # Extract data
    aen_epochs = [entry["epoch"] for entry in data["centralized_evaluate"]]
    aen_test_loss = [entry["test_loss"] for entry in data["centralized_evaluate"]]

    file_path = "/home/s7wu7/project/federated_fall_detection/src/fl-fall/outputs/2024-12-03/16-03-09/results.json"
    # Load JSON data from a file
    with open(file_path, 'r') as file:
        data = json.load(file)
    # Extract data
    fl_aen_epochs = [entry["round"] for entry in data["federated_evaluate"]]
    fl_aen_test_loss = [entry["federated_evaluate_loss"] for entry in data["federated_evaluate"]]

    file_path = "/home/s7wu7/project/federated_fall_detection/result/classification/SiSFall_results.json"
    # Load JSON data from a file
    with open(file_path, 'r') as file:
        data = json.load(file)
    # Extract data
    cl_epochs = [entry["epoch"] for entry in data["centralized_evaluate"]]
    cl_test_loss = [entry["test_loss"] for entry in data["centralized_evaluate"]]
    
    file_path = "/home/s7wu7/project/federated_fall_detection/result/classification/federated_SiSFall_results.json"
    # Load JSON data from a file
    with open(file_path, 'r') as file:
        data = json.load(file)
    # Extract data
    fl_cl_epochs = [entry["epoch"] for entry in data["centralized_evaluate"]]
    fl_cl_test_loss = [entry["test_loss"] for entry in data["centralized_evaluate"]]
        
    # Plotting
    fig = plt.figure(figsize=(16, 12))
    plots = fig.subplots(2, 1, sharex=True)
    plots[0].plot(aen_epochs, aen_test_loss,linewidth=5, label='Centeralized Autoencoder Test Loss')
    plots[0].plot(fl_aen_epochs, fl_aen_test_loss,linewidth=5, label='Federated Autoencoder Test Loss')
    plots[1].plot(cl_epochs, cl_test_loss,linewidth=5, label='Centeralized Classification Test Loss')
    plots[1].plot(fl_cl_epochs, fl_cl_test_loss,linewidth=5, label='Federated Classification Test Loss')
    
    # Adding title and labels
    # plots[0].set_title('Autoencoder and Classifier Test Loss Over Epochs')
    plots[1].set_xlabel('Epoch')
    plots[0].set_ylabel('MSE Loss')
    plots[1].set_ylabel('Crossentropy Loss')
    plots[0].legend(frameon=False)
    plots[1].legend(frameon=False)

    # Display the plot
    # plots[0].grid(True)
    # plots[1].grid(True)
    plt.tight_layout()
    plt.savefig(f"paper_loss.png", transparent=True)

def main():
    matplotlib.rcParams.update({'font.size': 32})
    files = [
        # "UpFall_results.json",
        # "federated_SiSFall_results.json",
        ("centeralized_MobiAct_results.json", None)
    ]
    for temp_file in files:
        cen_file, fed_file = temp_file
        # plot_test_losses()
        plot_test_metrics(centeralazied_file_path=cen_file, federated_file_path=fed_file)
        plot_reconstruction_training_result(cen_file)
        plot_classification_training_result(cen_file)
    
if __name__ == '__main__':
    main()