import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


def make_embedding_gif(
    npz_path: str,
    output_gif: str = "embeddings_trajectory.gif",
    labels: list = None,
    dpi: int = 120,
    fps: int = 4,
):
    """
    Creates a GIF visualizing embedding movement over epochs.

    Parameters
    ----------
    npz_path : str
        Path to `embedding_coords_history.npz` created during training.
        Must contain array 'coords' with shape (epochs, num_embeddings, 2).

    output_gif : str
        Output GIF filename.

    labels : list
        Optional list of time-series IDs used to annotate points.

    dpi : int
        Resolution for frames.

    fps : int
        Frames per second in the GIF.
    """

    # Load npz file
    data = np.load(npz_path)
    coords = data["coords"]  # shape: (E, N, 2)

    num_epochs, num_embeddings, _ = coords.shape

    # Labels fallback
    if labels is None:
        labels = [str(i) for i in range(num_embeddings)]

    # Prepare output frames folder
    frame_dir = "_embedding_frames"
    os.makedirs(frame_dir, exist_ok=True)

    frame_paths = []

    # Determine plot limits (fixed across epochs)
    # xmin = coords[:, :, 0].min()
    # xmax = coords[:, :, 0].max()
    # ymin = coords[:, :, 1].min()
    # ymax = coords[:, :, 1].max()
    xmin = -15
    xmax = 15
    ymin = -10
    ymax = 10

    # add padding to axes
    dx = 0.1 * (xmax - xmin + 1e-9)
    dy = 0.1 * (ymax - ymin + 1e-9)
    xmin -= dx
    xmax += dx
    ymin -= dy
    ymax += dy

    # Generate frames
    for epoch in range(num_epochs):

        fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)

        epoch_coords = coords[epoch]

        ax.scatter(epoch_coords[:, 0], epoch_coords[:, 1], s=50, alpha=0.8, color="C0")
        highlight_idx = 43
        ax.scatter(
            epoch_coords[highlight_idx, 0],
            epoch_coords[highlight_idx, 1],
            s=80,
            color="red",
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

        # Annotate
        for i in range(num_embeddings):
            ax.text(
                epoch_coords[i, 0],
                epoch_coords[i, 1],
                labels[i],
                fontsize=7,
                ha="center",
                va="center",
            )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"Embeddings in 2D – Epoch {epoch}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True)

        # Save frame
        frame_path = os.path.join(frame_dir, f"frame_{epoch:04d}.png")
        fig.savefig(frame_path, bbox_inches="tight")
        plt.close(fig)

        frame_paths.append(frame_path)

    # Create GIF
    print(f"Creating GIF: {output_gif}")
    with imageio.get_writer(output_gif, mode="I", fps=fps) as writer:
        for path in frame_paths:
            writer.append_data(imageio.imread(path))

    # Cleanup
    print("Cleaning up frame images...")
    for path in frame_paths:
        os.remove(path)
    os.rmdir(frame_dir)

    print(f"Done! GIF saved as {output_gif}")


# Example usage
if __name__ == "__main__":
    make_embedding_gif(
        npz_path="out/seed1/train80/experiment01_global_simple embedding scaled_normal/embeddings/embedding_coords_history.npz",
        output_gif="embedding_evolution.gif",
        labels=None,
        fps=2,
    )
