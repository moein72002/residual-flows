import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

def plot_in_out_histogram(hist_name, id_list_name, id_list, out_list_name, out_list, epoch):
    print(f"start calculating AUC in epoch {epoch}")
    print(f"id_list.shape: {id_list.shape}")
    print(f"out_list.shape: {out_list.shape}")
    anomaly_scores = np.array([])
    test_labels = np.array([])
    anomaly_scores = np.concatenate([1 - id_list, 1 - out_list], axis=0)
    test_labels = np.concatenate([np.zeros((id_list.shape[0],)), np.ones((out_list.shape[0],))], axis=0)
    print(f"anomaly_scores.shape: {anomaly_scores.shape}")
    print(f"test_labels.shape: {test_labels.shape}")
    auc = roc_auc_score(test_labels, anomaly_scores)
    print(f"auc in epoch {epoch}: {auc * 100}%")
    print(f"finish calculating AUC in epoch {epoch}")
    print(f"start plot_in_out_histogram in epoch {epoch}")
    # Plot histograms
    plt.hist(id_list, bins=100, alpha=0.5, color='blue', label=id_list_name)
    plt.hist(out_list, bins=100, alpha=0.5, color='orange', label=out_list_name)

    # Add labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of ID VS OUT {hist_name}')

    # Add legend
    plt.legend()

    # Show the plot
    save_file_name = f"./prob_results/probability_chart_in_epoch_{epoch}_auc_{auc * 100}.png"
    plt.savefig(save_file_name)
    plt.show()
    print(f"finish plot_in_out_histogram in epoch {epoch}")

# plot_in_out_histogram("ood", "cifar10", np.random.rand(200), "cifar100", np.random.rand(200) - 0.5, 0)

