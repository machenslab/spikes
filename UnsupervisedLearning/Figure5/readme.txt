The program "overnight" runs the learning algorithm on the speech
dataset, and plots fig 5. If you launch it, everything should run
smoothly, and it should take around 6 to 10 hours with a typical
computer.

If you prefer to use the already learnt weights (Icurrent) and just
retrain the network on the new stimulus, set the variable 'learnanew' 
to 1, then run fig5Anew. (takes 5 to 10 minutes)

If you prefer to use the already retrained weights, set 'learnanew'
to 0, then run fig5Anew.

Individual programs:

netall runs the learning algorithm on a random speech segment.

netrun runs the network on a random speech segment without
plasticity

clampall clamps all the neurons one by one (i.e. they are prevented
from firing) and measure their E and I currents.

netruncl runs the clamped network on any input stimulus.

netadapt re-train the network on arbitrary input signals. 

fig5Anew plots the same results as fig 5 (since a different
random seed is used each time, this will never be exactly like fig
5).


