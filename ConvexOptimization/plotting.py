import holoviews as hv
import numpy as np

# get colors
colors = hv.core.options.Cycle.default_cycles['default_colors']
Ncolors = len(colors)

def spike_plot(times, spikes, base_offset, offset):
    """Plots a set of neurons' spikes, given a 2d array of 0's and 1's.

    Parameters
    ----------
    times : array
        array of times
    spikes : array
        2D-array of 0's and 1's (1's being spikes),
        of size (n_cells, n_timepoints)
    base_offset : float
        y-axis offset of all spikes
    offset : float
        y-axis offset between each row of spikes

    Returns
    -------
    Holoviews Overlay
        An overlay with all the spikes shown
    """
    # make spike plot animation
    out = hv.Overlay()
    for i in range(spikes.shape[0]):
        spiketimes = times[np.where(spikes[i, :]==1)[0]]
        if len(spiketimes)>0:
            opts = hv.opts.Scatter(color=colors[i%len(colors)])
            out *= hv.Scatter(
                zip(spiketimes, np.ones(len(spiketimes))*offset*i+base_offset),
                              kdims='Time (s)',
                              vdims='Neuron', group='spikes').opts(opts)
        else:
            opts = hv.opts.Scatter(color='w', alpha=0)
            out *= hv.Scatter([],
                          kdims='Time (s)',
                          vdims='Neuron', group='spikes').opts(opts)
    return out

def plot_spikes_single(times, spikes, color, alpha=1, s=10, offset=0, base_offset=0):
    """Plots a single neuron's spikes.

    Parameters
    ----------
    times : array
        array of times
    spikes : array
        2D-array of 0's and 1's (1's being spikes),
        of size (n_cells, n_timepoints)
    color : string
        the color of the plotted spikes
    alpha : float
        the alpha of the plotted spikes (Between 0 and 1)
    s : int
        size of the plotted spikes
    offset : float
        y-axis offset between each row of spikes
    base_offset : float
        y-axis offset of all spikes

    Returns
    -------
    Holoviews Overlay
        An overlay with all the spikes shown
    """
    spiketimes = times[np.where(spikes==1)[0]]
    opts = hv.opts.Scatter(color=color, s=spikes, alpha=alpha)
    out = hv.Scatter(
        zip(spiketimes, np.ones(len(spiketimes))*offset+base_offset),
                      kdims='Time (s)',
                      vdims='Neuron', group='spikes2').opts(opts)
    return out