import numpy as np
import numba as nb


@nb.jit(nopython=True)
def run_snn_trial(x_sample,
                  F_weights,
                  omega,
                  thresholds,
                  dt,
                  leak,
                  mu=0.,
                  sigma_v=0.
                  ):
    """
    Function to simulate the spiking network for defined connectivity parameters, thresholds and time parameters.
    It returns the instantaneous firing rates of neurons for the whole simulation time
    Parameters
    ----------
    x_sample: array
        Input array (shape=[K, num_bins])
    F_weights: array
        Feed-forward weights (shape=[N, K])
    omega: array
        Recurrent weights (shape=[N, N])
    thresholds: array
        Neurons thresholds (shape=[N,])
    dt: float
        time step
    leak: float
        membrane leak time-constant
    mu: float
        controls spike cost
    sigma_v: float
        controls variance of voltage noise

    Returns
    -------
    array
        network instantaneous firing rates (shape=[N x num_bins])
    """

    # initialize system
    N = F_weights.shape[0]  # number of neurons
    num_bins = x_sample.shape[1]  # number of time bins
    firing_rates = np.zeros((N, num_bins))
    V_membrane = np.zeros(N)

    # implement the Euler method to solve the differential equations
    for t in range(num_bins - 1):
        # compute command signal
        command_x = (x_sample[:, t + 1] -
                     x_sample[:, t]) / dt + leak * x_sample[:, t]

        # update membrane potential
        V_membrane += dt * (-leak * V_membrane +
                            np.dot(F_weights, command_x)
                            ) + np.sqrt(2 * dt * leak) * sigma_v * np.random.randn(N)

        # update firing rates
        firing_rates[:, t + 1] = (1 - leak * dt) * firing_rates[:, t]

        # Check if any neurons are past their threshold during the last time-step
        diff_voltage_thresh = V_membrane - thresholds
        spiking_neurons_indices = np.arange(N)[diff_voltage_thresh >= 0]
        if spiking_neurons_indices.size > 0:
            # Pick the neuron which likely would have spiked first, by max distance from threshold
            to_pick = np.argmax(V_membrane[spiking_neurons_indices] - thresholds[spiking_neurons_indices])
            s = spiking_neurons_indices[to_pick]

            # Update membrane potential
            V_membrane[s] -= mu
            V_membrane += omega[:, s]

            # Update firing rates
            firing_rates[s, t + 1] += 1

        else:
            pass

    return firing_rates


@nb.jit(nopython=True)
def update_weights(x_sample,
                   y_target_sample,
                   F_weights,
                   G_weights,
                   omega,
                   thresholds,
                   buffer_bins,
                   dt,
                   leak,
                   leak_thresh,
                   alpha_thresh,
                   alpha_F,
                   mu=0.,
                   sigma_v=0.,
                   ):
    """
    Train the network in one trial with one presented input-target pair (x_sample, y_target_sample)
    The function returns the updated thresholds and feed-forward weights after that trial
    Parameters
    ----------
    x_sample: array
        Input array (shape=[K, num_bins])
    y_target_sample: array
        target sample (shape=[M,])
    F_weights: array
        Feed-forward weights (shape=[N, K])
    G_weights: array
        Encoder weights (shape=[N, M])
    omega: array
        Recurrent weights (shape=[N, N])
    thresholds: array
        Neurons thresholds (shape=[N,])
    buffer_bins: int
        Number of bins before learning starts
    dt: float
        time step
    leak: float
        membrane leak time-constant
    leak_thresh: float
        controls the speed of drift in thresholds
    alpha_thresh: float
        learning rate of thresholds (>> leak thresh)
    alpha_F: float
        learning rate for forward weights
    mu: float
        controls spike cost
    sigma_v: float
        controls variance of voltage noise
    Returns
    -------
    array
        updated thresholds array
    array
        updated feed-forward weights array
    """

    # initialize system
    N = F_weights.shape[0]
    num_bins = x_sample.shape[1]
    firing_rates = np.zeros((N, num_bins))
    V_membrane = np.zeros(N)

    # implement the Euler method to solve the differential equations
    for t in range(num_bins - 1):
        # compute command signal
        command_x = (x_sample[:, t + 1] -
                     x_sample[:, t]) / dt + leak * x_sample[:, t]

        # update membrane potential
        V_membrane += dt * (-leak * V_membrane +
                            np.dot(F_weights, command_x)
                            ) + np.sqrt(2 * dt * leak) * sigma_v * np.random.randn(N)
        # update rates
        firing_rates[:, t + 1] = (1 - leak * dt) * firing_rates[:, t]

        # Check if any neurons are past their threshold during the last time-step
        diff_voltage_thresh = V_membrane - thresholds
        spiking_neurons_indices = np.arange(N)[diff_voltage_thresh >= 0]
        if spiking_neurons_indices.size > 0:
            # Pick the neuron which likely would have spiked first, by max distance from threshold
            to_pick = np.argmax(V_membrane[spiking_neurons_indices] - thresholds[spiking_neurons_indices])
            s = spiking_neurons_indices[to_pick]

            # Update membrane potential
            V_membrane[s] -= mu
            V_membrane += omega[:, s]

            # Update rates with spikes
            firing_rates[s, t + 1] += 1

            # !! Update weights
            if t >= buffer_bins:
                proj_error_neuron = F_weights[s, :] @ x_sample[:, t] - thresholds[
                    s] - G_weights[s, :] @ y_target_sample

                dLdthresh = -proj_error_neuron
                dLdf_weights = proj_error_neuron * x_sample[:, t]

                thresholds[s] -= alpha_thresh * dLdthresh
                F_weights[s, :] -= alpha_F * dLdf_weights

        else:
            pass

        # drift thresholds
        if t >= buffer_bins:
            thresholds -= dt * leak_thresh

    return thresholds, F_weights


def run_snn(x,
            F_weights,
            omega,
            thresholds,
            dt,
            leak,
            mu=0.,
            sigma_v=0.,
            silence_T=None,
            silence_prop=0,
            delay=0
            ):
    """
    Function to simulate the spiking network for defined connectivity parameters, thresholds and time parameters.
    It returns the firing rates, voltages, currents, and spikes.

    Parameters
    ----------
    x: array
        Input array (shape=[K, num_bins])
    F_weights: array
        Feed-forward weights (shape=[N, K])
    omega: array
        Recurrent weights (shape=[N, N])
    thresholds: array
        Neurons thresholds (shape=[N,])
    dt: float
        time step
    leak: float
        membrane leak time-constant
    mu: float
        controls spike cost
    sigma_v: float
        controls variance of voltage noise
    silence_T: int
        From which time-point to silence neurons
    silence_prop: float
        Which proportion of the population to silence
    delay: int
        Synaptic delay in recurrent connections in number of timesteps

    Returns
    -------
    array
        network instantaneous firing rates (shape=[N x num_bins])
    array
        network spikes (shape=[N x num_bins])
    array
        network voltages (shape=[N x num_bins])
    array
        network excitatory currents (shape=[N x num_bins])
    array
        network inhibitory currents (shape=[N x num_bins])
    """
    # initialize system
    N = F_weights.shape[0]  # number of neurons
    num_bins = x.shape[1]  # number of time bins
    firing_rates = np.zeros((N, num_bins))
    spikes = np.zeros((N, num_bins))
    V_membrane = np.zeros((N, num_bins))
    I_E = np.zeros((N, num_bins))
    I_I = np.zeros((N, num_bins))

    # separate recurrent weights into inhibitory and excitatory
    omega_e, omega_i = omega.copy(), omega.copy()
    omega_e[omega < 0] = 0
    omega_i[omega > 0] = 0
    omega_i[range(N), range(N)] = 0  # remove self-resets
    omega_e[range(N), range(N)] = 0  # remove self-resets

    # separate feed-forward weights into positive and negative
    F_pos, F_neg = F_weights.copy(), F_weights.copy()
    F_pos[F_weights < 0] = 0
    F_neg[F_weights > 0] = 0

    # if not given, set silence point at end
    if silence_T is None:
        silence_T = num_bins + 1

    # implement the Euler method to solve the differential equations
    for t in range(num_bins - 1):
        # compute command signal
        command_x = (x[:, t + 1] -
                     x[:, t]) / dt + leak * x[:, t]

        # update membrane potential
        V_membrane[:, t + 1] = V_membrane[:, t] + dt * (-leak * V_membrane[:, t] +
                                                        np.dot(F_weights, command_x)
                                                        ) + np.sqrt(2 * dt * leak) * sigma_v * np.random.randn(N)
        if t >= delay:
            V_membrane[:, t + 1] = V_membrane[:, t + 1] + np.dot(omega_rec, spikes[:, t-delay])

        # get positive/negative inputs
        command_x_pos, command_x_neg = command_x.copy(), command_x.copy()
        command_x_pos[command_x < 0] = 0
        command_x_neg[command_x > 0] = 0

        # update currents
        dI_I = np.dot(F_neg, command_x_pos) + np.dot(F_pos, command_x_neg) # inhibitory inputs
        if t >= delay:
            dI_I += np.dot(omega_i, spikes[:, t-delay] / dt)
        I_I[:, t+1] = I_I[:, t] + dt*(-I_I[:, t]*leak + dI_I)
        dI_E = np.dot(F_pos, command_x_pos) + np.dot(F_neg, command_x_neg) # excitatory inputs
        if t >= delay:
            dI_E += np.dot(omega_e, spikes[:, t-delay] / dt)
        I_E[:, t+1] = I_E[:, t] + dt*(-I_E[:, t]*leak + dI_E)

        # update firing rates
        firing_rates[:, t + 1] = (1 - leak * dt) * firing_rates[:, t]

        # silence neurons
        if t > silence_T:
            V_membrane[int(N * (1 - silence_prop)):, t + 1] = -100

        # Check if any neurons are past their threshold during the last time-step
        diff_voltage_thresh = V_membrane[:, t + 1] - thresholds
        spiking_neurons_indices = np.arange(N)[diff_voltage_thresh >= 0]
        if spiking_neurons_indices.size > 0:
            if delay == 0:
                # Pick the neuron which likely would have spiked first, by max distance from threshold
                to_pick = np.argmax(diff_voltage_thresh[spiking_neurons_indices])
                s = spiking_neurons_indices[to_pick]

                # Update membrane potential
                V_membrane[s, t + 1] -= mu
                spikes[s, t + 1] = 1

                # Update firing rates
                firing_rates[s, t + 1] += 1
            else:
                # Update membrane potential
                V_membrane[spiking_neurons_indices, t + 1] -= mu
                # V_membrane[:, t + 1] += omega[:, s]
                spikes[spiking_neurons_indices, t + 1] = 1

                # Update firing rates
                firing_rates[spiking_neurons_indices, t + 1] += 1

        else:
            pass

    return firing_rates, spikes, V_membrane, I_E, I_I


@nb.jit(nopython=True)
def run_maxout(x_sample, F_weights, G_weights, thresholds):
    # calculate activities
    neural_boundary = np.zeros((G_weights.shape[1], G_weights.shape[0] + 1))
    neural_boundary[:, 1:] = F_weights @ x_sample / G_weights.ravel() - thresholds / G_weights.ravel()
    y_out = neural_boundary.max()
    nactive = neural_boundary.argmax()

    return y_out, nactive
