% This program produces Figure 2 for "Optimal Compensation for
% Neuron Loss" by David Barrett, Sophie Deneve, and Christian Machens
%
% This program simulates a heterogenoues network of spiking neurons
% with separate E and I populations, representing a single signal
% All free parameters can be changed from their default values 
% (within limits - note that plotting is not optimized for large changes)

clear all;
rng(12345);                      % initialize Random Number Generator

%==========================  FIGURE 2 ======================================

%--------------------------- FREE PARAMETERS -------------------------------

% Neurons
NE = 80;                         % Number of excitatory neurons
NI = 20;                         % Number of inhibitory neurons
N = NE+NI;                       % Total number of neurons
% needless to say, the following indices should not exceed N
Nko1 = 1:60;                     % indices of neurons in first k.o.
Nko2 = 86:100;                   % indices of neurons in second k.o.
neuron = 70;                     % highlighted neuron (in Panel E/F)

% Decoder (note that all entries need to be positive !)
DE = (2+0.2*randn(1,NE))/NE;     % Decoder for E population
DI = (0.3+0.003*randn(NE,NI))/NI;% Decoder for I population

% Other parameters
BetaE=0.8/NE^2;                  % Quadratic cost for E population
BetaI=0.2/NI^2;                  % Quadratic cost for I population
lam = 0.05;                      % Decoder time scale lambda (in deca-Hz)
sigV = 0.01/N;                   % stdev of voltage noise

%--------------------------- TIME PARAMETERS -------------------------------

% time steps and knock-out schedule
Time= 500;                       % Simulation time in centi-sec
dt=0.005;                        % Euler time step in centi-sec
t = 0:dt:Time-dt;                % array of time points
Nt=Time/dt;                      % Number of time steps
Timeon = 40;                     % Burn in time (time from which we plot)
tko1 = 300;                      % onset time of first k.o.
tko2 = 400;                      % onset time of second k.o.

%--------------------------- INPUT SIGNALS ---------------------------------

x=zeros(1,Nt);                   % Initialise Signal ...
x(1:(Timeon +40)/dt)=0;
x((Timeon +40)/dt+1:(Timeon +100)/dt)=0.24;
x((Timeon +100)/dt:(Timeon+200)/dt)=0.48;
x((Timeon +200)/dt:Nt)=0.36;
x=2*smooth( x, 500 );            % Smooth signal
dxdt(1,:) = [0,diff(x)]/dt;      % Compute signal derivative
c = lam*x + dxdt;                % Actual input into network

%--------------------------- DERIVED PARAMETERS ----------------------------

% Inhibitory Connectivity
HII = DI'*DI + BetaI*eye(NI);    % Eq. (32) - auxiliary short-hand
RI = diag( HII );                % Eq. (36) - I Resets
OmII = HII - diag( RI );         % Eq. (33) - I->I connections
OmIE = DI';                      % Eq. (34) - E->I connections
TI = diag( HII ) / 2;            % Eq. (35) - thresholds

% Excitatory connectivity
HEE = ( DE'*DE + BetaE*eye(NE) );% Eq. (40) - auxiliary short-hand
RE = diag(HEE);                  % Eq. (45) - E Resets
FE = DE';                        % Eq. (41) - FF connections
OmEE = thlin( -HEE );            % Eq. (42) - E->E connections
OmEI = (thlin( HEE ) - diag(RE))*DI; % Eq. (43) - I->E connections 
TE = diag(HEE)/2;                % Eq. (44) - E thresholds

% Combining all matrices into one for faster simulation
T = [TE; TI];                    % all thresholds
F = [FE; zeros(NI,1)];           % full feedforward connectivity
Om = [OmEE-diag(RE), -OmEI; OmIE, -OmII-diag(RI)]; % full recurrent connectivity
Omrecurrent = (Om - diag(diag(Om))); % connectivity without self-resets

%=========================== SIMULATION START ==============================

% Simulation variables and initial conditions
V = zeros( N, Nt);               % membrane voltages
s = zeros( N, Nt);               % spike trains
r = zeros( N, Nt);               % instantaneous firing rates
Vpos = zeros(1,Nt);              % positive inputs into example neuron
Vneg = zeros(1,Nt);              % negative inputs into example neuron

% Simulate network
for k=2:Nt
  
  % Voltage and firing rate update with Euler method
  dVdt        = -lam*V(:,k-1) + F*c(k-1) + Om*s(:,k-1);
  drdt        = -lam*r(:,k-1) + s(:,k-1);
  V(:,k)      = V(:,k-1) + dVdt*dt + sigV*randn(N,1).*sqrt(dt); 
  r(:,k)      = r(:,k-1) + drdt*dt;
  
  % knock-out neurons
  if t(k)>tko1, V(Nko1,k) = 0; end;
  if t(k)>tko2, V(Nko2,k) = 0; end;
  
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
  
  % compute positive / negative inputs into example neuron
  allsynapticinputs = [F(neuron,:) .* x(:,k-1); Omrecurrent(neuron,:)'.*r(:,k-1)];
  Vpos(k) = sum( thlin( allsynapticinputs) );
  Vneg(k) = sum( -thlin( -allsynapticinputs) );
end

% Separate results according to E/I population
rE = r(1:NE,:);                  % rates from E population
rI = r(NE+1:end,:);              % rates from I population
VE = V(1:NE,:);                  % voltages of E population
VI = V(NE+1:end,:);              % voltages of I population

xest = DE*rE;                    % signal estimate from E population
rEest = DI*rI;                   % E ouput estimate from I population

%% ========================= SIMULATION END ================================



%% ------------------------- PLOT RESULTS ----------------------------------

figure(2); clf;
set( gcf, 'Color', 'w' );
ncolor  = [0 0.7 0];             % color of highlighted neuron
ncolexc = [0.7 1 0];             % color of highlighted neuron (exc currents)
ncolinh = [0 0.5 0];             % color of highlighted neuron (inh currents)

%% ------------------------- FIG 2D, stimulus and estimate -----------------
axes('pos',[0.1  0.78 0.8 0.18 ]);
hold on;
plot(t, xest(1,:),'LineWidth', 3, 'Color', [0.3 0.6 1]);
plot(t, x(1,:),'k-','LineWidth', 1)
ylabel('Signal (a.u.)');
box off;
set( gca,'XTick',[], 'XColor','w');
set( gca,'YTick', [0 0.33 0.66 1] );
set( gca,'YTickLabel', {'','', '0', '', '', ''}, 'TickDir', 'out');
axis([Timeon Time -0.2 1.2 ]);

%% ------------------------- FIG 2E: raster plot ---------------------------
axes('pos',[0.1 0.4 0.8 0.34 ]);
hold on;
scop = s;
scop(neuron,:)=-scop(neuron,:);
i    = find(scop>0);                   % linear indices of all (-'neuron') spikes
isel = find(scop<0);                   % linear indices of 'neuron's spikes
[yind,tind] = ind2sub(size(s),i);      % matrix indices from linear indices
[ysind,tsind] = ind2sub(size(s),isel);
yind(yind>NE) = yind(yind>NE) + 5;     % offset inhibitory neurons
ysind(ysind>NE) = ysind(ysind>NE) + 5;
plot( t(tind), yind,'.k', 'MarkerSize', 5);
plot( t(tsind), ysind,'.', 'MarkerSize', 16, 'Color', ncolor)
boxcol = [0.9 0.9 0.9];
pp = patch( [Timeon-5.5 Timeon+5.5 Timeon+5.5 Timeon-5.5 Timeon-5.5], [NE+5 ...
                    NE+5 N+5 N+5 NE+5], 'r' );
set( pp, 'FaceColor', boxcol, 'EdgeColor', boxcol );
text( Timeon+1,NE+5+NI/2,'I');
boxcol = [0.65 0.65 0.65];
pp = patch( [Timeon-5.5 Timeon+5.5 Timeon+5.5 Timeon-5.5 Timeon-5.5], [0 0 NE NE 0], 'r');
set( pp, 'FaceColor', boxcol, 'EdgeColor', boxcol );
text( Timeon,NE/2,'E');
axis([Timeon Time 0 N+5]);
axis off;
text( Timeon-15, N/2, 'Spike trains', 'Rotation', 90, ...
'HorizontalAlignment','Center','VerticalAlignment','Middle');
set( gca, 'TickDir', 'out');

%% ------------------------- FIG 2F: voltage of example neuron -------------
axes('pos',[0.1 0.26 0.8 0.1 ]);
hold on;
VT=T(neuron);                % voltage threshold of neuron
thisV = V(neuron,:) + s(neuron,:)*0.8*max(max(V))*dt; % add spikes
plot(t,thisV,'-','LineWidth', 0.5, 'Color', ncolor);
plot([0,Time],[VT,VT],'--k','LineWidth', 1);
plot([0,Time],[-VT,-VT],'--k','LineWidth', 1);
axis([Timeon Time -3*VT  3*VT]);
set( gca,'XTick',[], 'XColor','w');
set( gca,'YTick', [-VT 0 VT 2*VT] );
set( gca,'YTickLabel', {'R', '', 'T', ''}, 'TickDir', 'out');
ylabel('Voltage (a.u)');

%% ------------------------- FIG 2G: EI currents ---------------------------
axes('pos',[0.1 0.04 0.8 0.18 ]);
hold on;
plot([t(1) t(Nt)],[0 0],'k-','LineWidth', 1);
plot(t,Vpos,'-', 'LineWidth', 1.5, 'Color', ncolexc );
plot(t,Vneg,'-', 'LineWidth', 1.5, 'Color', ncolinh );
plot(t,V(neuron,:),'-','LineWidth', 1.5, 'Color', ncolor );
ysize=max(max(Vpos),(-1)*(min(Vneg)));
axis([Timeon Time -ysize ysize]);
set( gca,'XTick',[], 'XColor','w');
set( gca,'YTick', [-ysize -ysize/2 0 ysize/2 ysize] );
set( gca,'YTickLabel', {'', '', '0', '', ''}, 'TickDir', 'out');
ylabel('Current (a.u)');
hold on;                     
% insert blow up (part 1)
t1 = 27000; t2 = 28000;
plot( [t(t1) t(t2) t(t2) t(t1) t(t1)], ...
      [-ysize -ysize ysize ysize -ysize]/10, 'k-' );
plot( [t(t2) t(t2+5500)], [ysize/10 ysize/2], 'k-' );
pos = get( gca, 'pos' );     
% insert scale bar
scl = (pos(3))/(Time-Timeon);
scalebar([0.9-scl*55, 0.04, scl*50, 0.015], '500 msec');
% blow up (part 2)
axes('pos', [0.32 0.152 0.08 0.05] );
plot( t(t1:t2), V(neuron,t1:t2), '-', 'Color', ncolor );
axis( [t(t1) t(t2) -ysize/32 ysize/32] );
set( gca, 'YTick', [-ysize/32, 0, ysize/32], ...
          'YTickLabel', {'', '0', ''}, 'XTick', [] );
set( gca, 'TickDir', 'out', 'TickLength', [0.05 0.05]);

%% ------------------------- KNOCK OUT LINES -------------------------------
axes('pos',[0 0 1 1]);
hold on;
xpos=0.555;
plot( [xpos xpos], [0.04 0.94], 'r--', 'Linewidth', 1);
annotation('arrow', [xpos xpos], [0.985 0.95], 'Color', 'r', ...
           'Linewidth', 2 );
axis([0 1 0 1]);
set( gca, 'Visible', 'off' );
set( gca, 'Color', 'none' );

% plot knock-out line 2
axes('pos',[0 0 1 1]);
hold on;
xpos=0.725;
plot( [xpos xpos], [0.04 0.94], 'r--', 'Linewidth', 1);
annotation('arrow', [xpos xpos], [0.985 0.95], 'Color', 'r', ...
           'Linewidth', 2 );
axis([0 1 0 1]);
set( gca, 'Visible', 'off' );
set( gca, 'Color', 'none' );
