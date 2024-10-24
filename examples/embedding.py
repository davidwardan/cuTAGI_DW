import numpy as np


class TimeSeriesEmbeddings:
    """
    Class to handle embedding operations with mean and variance.
    """

    def __init__(self, embedding_dim: tuple, encoding_type: str = None):
        self.embedding_dim = embedding_dim
        if encoding_type == "normal":
            self.mu_embedding = np.random.randn(*embedding_dim) / (embedding_dim[1])
            self.var_embedding = np.full(embedding_dim, 1.0)
        elif encoding_type == "onehot":
            epsilon = 1e-6
            if embedding_dim[0] != embedding_dim[1]:
                raise ValueError("Dimensions of the embedding must be equal.")
            self.mu_embedding = np.full(embedding_dim, epsilon)
            np.fill_diagonal(self.mu_embedding, 1.0)
            self.var_embedding = np.ones(embedding_dim)
        else:
            self.mu_embedding = np.full(embedding_dim, 1 / (embedding_dim[1]))
            self.var_embedding = np.ones(embedding_dim)

    def update(self, idx: int, mu_delta: np.ndarray, var_delta: np.ndarray):
        self.mu_embedding[idx] = self.mu_embedding[idx] + mu_delta
        self.var_embedding[idx] = self.var_embedding[idx] + var_delta

    def get_embedding(self, idx: int) -> tuple:
        return self.mu_embedding[idx], self.var_embedding[idx]

    def save(self, out_dir: str):
        np.savetxt(out_dir + "/embeddings_mu.csv", self.mu_embedding, delimiter=",")
        np.savetxt(out_dir + "/embeddings_var.csv", self.var_embedding, delimiter=",")

# TODO: Create a EmbeddingHandler class to handle the embeddings (build, reduce, input)
def build_vector(x: int, num_features_len: int, embedding_dim: int) -> np.ndarray:
    vector = np.zeros(x)
    cycle_length = num_features_len + embedding_dim

    # Iterate through the vector in steps of the cycle length
    for i in range(0, x, cycle_length):
        # Find the starting position of the embedding section in the current cycle
        embedding_start = i + num_features_len

        # Ensure the embedding section doesn't go out of bounds
        if embedding_start < x:
            # Set the values of the embedding section to ones
            end_position = min(embedding_start + embedding_dim, x)
            vector[embedding_start:end_position] = np.ones(
                end_position - embedding_start
            )

    return vector


def reduce_vector(x: np.ndarray, vector: np.ndarray, embedding_dim: int) -> np.ndarray:
    x = (x + 1) * vector
    x = x[x != 0] - 1  # remove zeros and reset index
    return x.reshape(-1, embedding_dim)


def input_embeddings(x, embeddings, num_features, embedding_dim):
    """
    Reads embeddings into the input vector.
    """
    x_var = x.copy()
    counter = 0
    last_idx = 0

    for item in x:
        if counter % num_features == 0 and counter != 0 and counter + last_idx < len(x):
            idx = int(x[counter + last_idx])
            embed_x, embed_var = embeddings.get_embedding(idx)
            (
                x[counter + last_idx : counter + last_idx + embedding_dim],
                x_var[counter + last_idx : counter + last_idx + embedding_dim],
            ) = (embed_x.tolist(), embed_var.tolist())
            last_idx = counter + embedding_dim + last_idx
            counter = 0
        else:
            counter += 1

    return np.array(x, dtype=np.float32), np.array(x_var, dtype=np.float32)
