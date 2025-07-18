import numpy as np


class TimeSeriesEmbeddings:
    """
    Class to handle embedding operations with mean and variance.
    """

    def __init__(
        self, embedding_dim: tuple, encoding_type: str = None, seed: int = None
    ):

        if seed is not None:
            np.random.seed(seed)
        self.embedding_dim = embedding_dim
        if encoding_type == "normal":
            self.mu_embedding = np.random.randn(*embedding_dim)
            self.var_embedding = np.full(embedding_dim, 1.0)
        elif encoding_type == "onehot":
            epsilon = 1e-6
            if embedding_dim[0] != embedding_dim[1]:
                raise ValueError("Dimensions of the embedding must be equal.")
            self.mu_embedding = np.full(embedding_dim, epsilon)
            np.fill_diagonal(self.mu_embedding, 1.0)
            self.var_embedding = np.ones(embedding_dim)
        elif encoding_type == "uniform":
            self.mu_embedding = np.full(embedding_dim, 1 / (embedding_dim[1]))
            self.var_embedding = np.ones(embedding_dim)
        elif encoding_type == "sphere":
            radius = 3  # radius of the sphere
            # sample means from a normal distribution
            vecs = np.random.randn(*embedding_dim)
            # project each row to the unit sphere
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            # scale to radius R
            self.mu_embedding = radius * vecs
            # small, fixed variance
            self.var_embedding = np.full(embedding_dim, 1.0)
        else:
            self.mu_embedding = np.full(embedding_dim, 1.0)
            self.var_embedding = np.ones(embedding_dim)

    def update(self, idx: int, mu_delta: np.ndarray, var_delta: np.ndarray):
        self.mu_embedding[idx] = self.mu_embedding[idx] + mu_delta
        self.var_embedding[idx] = self.var_embedding[idx] + var_delta

    def get_embedding(self, idx: int) -> tuple:
        return self.mu_embedding[idx], self.var_embedding[idx]

    def save(self, out_dir: str):
        np.savetxt(out_dir + "/embeddings_mu.csv", self.mu_embedding, delimiter=",")
        np.savetxt(out_dir + "/embeddings_var.csv", self.var_embedding, delimiter=",")


def build_vector(x_len: int, num_features_len: int, embedding_dim: int) -> np.ndarray:
    """
    Build a vector of length x_len, where each cycle consists of num_features_len zeros followed by embedding_dim ones.
    """
    cycle = num_features_len + embedding_dim
    # Create indices 0..x_len-1, take modulo to get position within each cycle
    idx = np.arange(x_len) % cycle
    # Positions >= num_features_len are embedding slots
    return (idx >= num_features_len).astype(float)


def reduce_vector(x: np.ndarray, vector: np.ndarray, embedding_dim: int) -> np.ndarray:
    x = (x + 1) * vector
    x = x[x != 0] - 1  # remove zeros and reset index
    return x.reshape(-1, embedding_dim)


def input_embeddings(x, embeddings, num_features, embedding_dim):
    """
    Reads embeddings into the input vector without explicit loops.
    """
    x = np.array(x, copy=False)
    x_var = x.copy()

    # block length: num_features data points + embedding_dim slots
    block_len = num_features + embedding_dim
    # start positions of embedding slots within x
    starts = np.arange(num_features, x.size, block_len)

    # the embedding indices are stored at the first position of each slot
    idxs = x[starts].astype(int)

    # gather all embeddings in one go using advanced indexing
    # embed_means = embeddings.mu_embedding[idxs]  # shape (n_blocks, embedding_dim)
    # embed_vars = embeddings.var_embedding[idxs]  # shape (n_blocks, embedding_dim)
    embed_means = embeddings[0]
    embed_vars = embeddings[1]

    # compute all target positions for the embedding slots
    offsets = np.arange(embedding_dim)
    positions = (starts[:, None] + offsets[None, :]).ravel()

    # assign the gathered embeddings into x and x_var
    x.flat[positions] = embed_means.ravel()
    x_var.flat[positions] = embed_vars.ravel()

    return x.astype(np.float32), x_var.astype(np.float32)
