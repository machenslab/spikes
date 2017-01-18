% OPTIMAL COMPENSATION FOR NEURON LOSS
%
% These programs allow the user to reproduce and manipulate the 
% Figures 1-6 of the paper "Optimal Compensation for Neuron Loss" by 
% David Barrett, Sophie Deneve, and Christian Machens
%
% Note: Figure7 and Figure89 require the function quadprog from the
% MATLAB optimization toolbox.
%
%=================================================================
%
% Figures
%   Figure1          - Figure 1, Panel C
%   Figure2          - Figure 2, Panels D - G
%   Figure3          - Figure 3, Panels D - I
%   Figure3JK        - Figure 3, Panels J, K
%   Figure4          - Figure 4, Panels A - C
%   Figure5          - Figure 5, Panels A - K
%   Figure6          - Figure 6, Panels B, D
%   Figure7          - Figure 7, Panels A - E 
%   Figure8_9        - Figure 8, Panels A - E, Figure 9B
%
% Auxiliary functions
%   scalebar         - Draws a scale bar into plot
%   smooth           - smoothes a time series
%   rbcolors         - rainbow color map
%   thlin            - threshold linear function
%   gabor_fn         - gabor function
%   orplot           - tuning curve plot in polar coordinates
%
% Data files
%   Weights.mat      - decoder weights for V1 example
%   img18.imc        - image for Figure 7
%   Figure3JK_sim.mat - temporary data storage
