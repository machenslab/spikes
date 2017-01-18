% This program produces Figure 1C for "Optimal Compensation for
% Neuron Loss" by David Barrett, Sophie Deneve, and Christian Machens

% This program simulates a homogeneous network of spiking neurons. 
% All free parameters can be changed from their default values 
% (within limits, since plotting is not optimized for large changes)

rng('default');

%--------------------------- FREE PARAMETERS -------------------------------

N    = 2;                        % Number of neurons in network
Nko  = 2;                        % indices of neurons to be k.o.
lam  = 0.1;                      % Decoder timescale (in inverse milliseconds)
QBeta = 0.05;                    % Quadratic firing rate cost
sigV = 0.001;                    % standard deviation of voltage noise
D = ones(1,N) + 0*randn(1,N);    % Decoder (homogeneous by default)

%--------------------------- SIMULATION AND DERIVED PARAMETERS -------------

% Connectivity
QBeta = QBeta / N^2;             % scale costs according to network size
Om=D'*D + QBeta*eye(N);          % Initialise Recurrent connectivity
T=diag(Om)/2;                    % thresholds

% Time steps (Euler method)
Time = 100;                      % Simulation time in milliseconds
tko  = 60;                       % time of knockout
dt = 0.05;                       % Time steps in milliseconds
t = 0:dt:Time-dt;                % array of time points
Nt=length(t);                    % Number of time steps

% Input signal
xsignal = 300;               
x=zeros(1,Nt);                   % Initialise Signal ...
x(Nt/8+1:(Nt-2)/3)=xsignal/200;
x((Nt-2)/3+1:Nt)=xsignal/100;
x = smooth( x, Nt/50 );          % smooth away the steps
dxdt = [0,diff(x)]/dt;           % Compute signal derivative
c = lam*x + dxdt;                % actual input into network

%--------------------------  SIMULATION ------------------------------------

% initial conditions
V = zeros(N,Nt);                 % voltages
s = zeros(N,Nt);                 % Spike trains
r = zeros(N,Nt);                 % Filtered spike trains (or firing rates)

% Simulate network
for k=2:Nt
  
  % Voltage and firing rate update with Euler method
  dVdt     = -lam*V(:,k-1) + D'*c(:,k-1) - Om*s(:,k-1);
  drdt     = -lam*r(:,k-1) + s(:,k-1);
  V(:,k)   = V(:,k-1) + dVdt*dt + sigV*randn(N,1).*sqrt(dt); 
  r(:,k)   = r(:,k-1) + drdt*dt;
  
  % knock-out neuron after time point 'tko'
  if t(k)>tko, V(Nko,k) = 0; end;
  
  % check threshold crossings; only one neuron should spike per
  % time step (this is a numerical trick which allows us to use
  % a larger time step; in general, the network becomes sensitive 
  % to delays if the redundancy grows, and the input dimensionality
  % remains small and fixed)
  spiker  = find( V(:,k) > T);
  Nspiker = length(spiker);
  if Nspiker>0,
    chosen_to_spike=spiker(randi(Nspiker)); 
    s(chosen_to_spike,k)=1/dt;
  end

end
xest = D*r;                      % compute readout with original decoder

%=========================== FIGURE 1C =====================================

figure(1); clf;
set(gcf, 'Color', 'w');

TimeEnd = Time-12;               % Right range of plots

% plot stimulus and estimate
axes('pos',[0.1 0.6 0.8 0.35 ] );
hold on;
plot( t, x,   '-k','LineWidth', 1.5);
plot( t, xest,'-','LineWidth', 1.5);
set( gca,'XTick',[], 'XColor','w');
set( gca,'YTick', [0 2 4],'YTickLabel', {'0', '', '4'}, 'TickDir', 'out' );
set( gca, 'LineWidth', 0.75);
ylabel( 'Signal (a.u.)');
axis( [0 TimeEnd -0.1 4 ]);

h = 0.32/N;
hs = 0.4/N;
v = 0.1:hs:(0.1+(N-1)*hs);

% plot voltage traces
for k=1:N
  axes('pos',[0.1 v(k) 0.8 h ] );
  spikesize=1.6*T(k);
  plot( t, V(k,:)+s(k,:)*spikesize*dt,'-k','LineWidth', 1.5);
  axis([0 TimeEnd -T(k) T(k)+spikesize/0.8]);
  text(-4,-0.1, sprintf( 'V_{%d}', k ) );
  axis off;
end
pos = get( gca, 'pos' );
scl = pos(3)/TimeEnd;
scalebar([0.9-scl*15, 0.08, scl*10, 0.01], '10 msec');

