import os
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def display_worst_regression_cases(df, dataset, days_scale, output_dir, max_n, epoch):
    n = min(max_n, len(df))

    # Make directories
    worst_cases_dir = os.path.join(output_dir, f"worst_{n}_cases")
    os.makedirs(worst_cases_dir, exist_ok=True)
    epoch_worst_cases_dir = os.path.join(worst_cases_dir, f"{epoch}")
    os.makedirs(epoch_worst_cases_dir, exist_ok=True)

    sorted_df = df.sort_values("error", ascending=False)

    for i in range(n):
        row = sorted_df.iloc[[i]]
        real_index = row.index.values[0]
        photo_id = row["photo_path"].values[0]

        data = dataset[real_index]
        x = torch.tensor(data["input"])
        y = data["age"]

        y_pred = row["y_pred"].values[0]
        y = np.round(y * days_scale)

        plt.imshow(x.permute(1, 2, 0))
        plt.title(f"Predicted: {y_pred}, Real: {y}, Error: {abs(y - y_pred)}")
        plt.savefig(os.path.join(epoch_worst_cases_dir, f"{i}_{photo_id}.png"))
        plt.close()


def display_predictions(predictions, output_path="regression_performance"):
    fig = plt.figure(figsize=(16, 10))
    ys = list(predictions.keys())
    y_max = np.max(ys)

    plt.plot([0, y_max], [0, y_max])

    for y, values in predictions.items():
        size = len(values)
        y = [y] * size
        plt.scatter(y, values)

    plt.savefig(f"{output_path}.png")
    plt.close()

    return fig


def compute_projection(model, dataloader, device):
    model.eval()
    projections = []
    projections_target_age = []
    with torch.no_grad():
        for inputs, targets_age in tqdm(dataloader, desc="Computing dataset projection"):
            inputs = inputs.to(device)
            projected = model(inputs).cpu()
            projections.append(projected)
            projections_target_age.append(targets_age.cpu())
    projections = torch.cat(projections)
    projections_target_age = torch.cat(projections_target_age)
    return projections.numpy(), projections_target_age.numpy()


def display_latent_space(projections, targets, color_map="jet"):
    projected_data = projections

    # Perform PCA on the data
    pca = PCA(n_components=2, random_state=42)
    projected_data = pca.fit_transform(projections)

    # Visualizing with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embedding = tsne.fit_transform(projections)

    # Set up the figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot the data points with color corresponding to age
    sc = axes[0].scatter(projected_data[:, 0], projected_data[:, 1], c=targets, cmap=color_map)
    cbar = plt.colorbar(sc)
    cbar.set_label("Age")

    # Add axis labels and title
    axes[0].set_xlabel("Principal Component 1")
    axes[0].set_ylabel("Principal Component 2")
    axes[0].set_title("PCA with Age")

    sc = axes[1].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=targets, cmap=color_map)
    axes[1].set_title("t-SNE Visualization")
    axes[1].set_xlabel("Dimension 1")
    axes[1].set_ylabel("Dimension 2")
    cbar = plt.colorbar(sc)
    cbar.set_label("Age")

    # Show the plot
    plt.show()
    return projected_data, pca


def plot_latent_space(model, dataloader, device):
    projections, projections_target = compute_projection(model, dataloader, device)
    pca_proj, pca = display_latent_space(projections, projections_target)
