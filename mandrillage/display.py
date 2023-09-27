import os
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def display_predictions(
    predictions, std_by_value, mean_by_value, output_path="regression_performance"
):
    # Create a figure with margin function
    fig = plt.figure(figsize=(16, 10))
    ys = list(predictions.keys())
    std = list(std_by_value.values())
    means = list(mean_by_value.values())

    y_max = np.max(ys)
    plt.plot([0, y_max], [0, y_max])

    data = zip(ys, std, means)
    data = sorted(data, key=lambda x: x[0], reverse=True)
    ys, std, means = zip(*data)

    plt.plot(ys, means)

    std = np.array(std)
    means = np.array(means)

    plt.fill_between(ys, means - std, means + std)

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
        for inputs, targets_age in tqdm(
            dataloader, desc="Computing dataset projection"
        ):
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
    sc = axes[0].scatter(
        projected_data[:, 0], projected_data[:, 1], c=targets, cmap=color_map
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("Age")

    # Add axis labels and title
    axes[0].set_xlabel("Principal Component 1")
    axes[0].set_ylabel("Principal Component 2")
    axes[0].set_title("PCA with Age")

    sc = axes[1].scatter(
        tsne_embedding[:, 0], tsne_embedding[:, 1], c=targets, cmap=color_map
    )
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
