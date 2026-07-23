"""Demonstrate smoothing fewer timesteps than the allocated capacity."""

import numpy as np

from pytagi.nn import SLSTM, OutputUpdater, Sequential, SLinear


def smooth_sequence(net, updater, inputs, observations):
    observation_variance = np.array([0.1], dtype=np.float32)

    for timestep in range(len(inputs)):
        net(inputs[timestep])
        updater.update(
            output_states=net.output_z_buffer,
            mu_obs=observations[timestep : timestep + 1],
            var_obs=observation_variance,
            delta_states=net.input_delta_z_buffer,
        )
        net.backward()

    return net.smoother()


def main():
    capacity = 8

    net = Sequential(SLSTM(2, 4), SLinear(4, 1))
    net.num_samples = capacity
    updater = OutputUpdater(net.device)

    first_inputs = np.array(
        [[0.0, 0.5], [0.5, 1.0], [1.0, 1.5]], dtype=np.float32
    )
    first_observations = np.array([0.25, 0.75, 1.25], dtype=np.float32)
    first_mean, first_variance = smooth_sequence(
        net, updater, first_inputs, first_observations
    )

    assert first_mean.shape == (1, 3)
    assert first_variance.shape == (1, 3)

    # Reuse the same allocated buffer for a shorter sequence. The unused tail
    # from the first sequence must not appear in the second smoother output.
    second_inputs = np.array([[1.5, 2.0], [2.0, 2.5]], dtype=np.float32)
    second_observations = np.array([1.75, 2.25], dtype=np.float32)
    second_mean, second_variance = smooth_sequence(
        net, updater, second_inputs, second_observations
    )

    assert second_mean.shape == (1, 2)
    assert second_variance.shape == (1, 2)

    print(f"Allocated capacity: {capacity}")
    print(f"First sequence timesteps: {first_mean.shape[1]}")
    print(f"Second sequence timesteps: {second_mean.shape[1]}")
    print("Second smoothed mean:", second_mean)


if __name__ == "__main__":
    main()
