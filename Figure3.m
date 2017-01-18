% This program produces Figure 3 for "Optimal Compensation for
% Neuron Loss" by David Barrett, Sophie Deneve, and Christian Machens
%
% This program simulates a heterogenoues network of spiking neurons
% representing two sinusoidal signals.
% All free parameters can be changed from their default values 
% (within limits - note that plotting is not optimized for large changes)

clear all;
rng(12345);                      % initialize Random Number Generator

%==========================  FIGURE 3 ======================================

%--------------------------- FREE PARAMETERS -------------------------------

% Neurons
N = 32;                          % Number of Neurons in Network
                                 % (should be a multiple of four !)
Nko1 = N+1-N/4:N;                % knocked-out neurons in first round
Nko2 = N+1-N/2:N;                % knocked-out neurons in second round
neuron  = [1,10,16];             % example neurons (do not exceed N)

% Decoder
D(:,1:N) = [ sin( 2*pi*(1:N)/N ); cos( 2*pi*(1:N)/N )] / N;

% Other parameters
QBeta=0.05/N^2;                  % Quadratic cost
LBeta=0.15/N^2;                  % Linear cost
lam = 0.1;                       % lambda (1/lambda measured in 10^-2 sec)
sigV = 0.05/N^2;                 % standard deviation of voltage noise

%--------------------------- TIME PARAMETERS -------------------------------

Time= 1000;                      % Simulation time in 10^-2 sec
dt=0.01;                         % Euler time step in 10^-2 sec
t = 0:dt:Time-dt;                % array of time points
Nt=length(t);                    % Number of time steps
Timeon = 250;                    % Burn-in time (when plotting starts)
Ncycles = 4;                     % Number of sine wave sweeps;
Tcycle = Time/Ncycles;           % Time for each sweep;
tko1 = 2*Tcycle;                 % time of first knock-out
tko2 = 3*Tcycle;                 % time of second knock-out

%--------------------------- INPUT -----------------------------------------

f = Ncycles*2*pi/Time;           % Initialise Signal ...
x = [-sin(f*t); cos(f*t)];
dxdt = [ [0;0], diff(x')']/dt;   % Compute derivative
c = lam*x + dxdt;                % actual input into network

%--------------------------- DERIVED PARAMETERS ----------------------------
                                 
Om=D'*D + QBeta*eye(N);          % Initialize Recurrent connectivity
T=(diag(Om)+QBeta+LBeta)/2;      % Thresholds
Omrecurrent = Om - diag(diag(Om)); % Connectivity w/o self-connections


%=========================== SIMULATION START ==============================

% Simulation variables and initial conditions
V      = zeros(N,Nt);
s      = zeros(N,Nt);
r      = zeros(N,Nt);

% Highlighted example neurons and their E I currents
Vpos   = zeros(length(neuron),Nt); % w/o self-reset
Vneg   = zeros(length(neuron),Nt);
VRpos  = zeros(length(neuron),Nt); % with self-reset
VRneg  = zeros(length(neuron),Nt);

% Simulate network
for k=2:Nt
  
  % Voltage and firing rate update with Euler method
  dVdt     = -lam*V(:,k-1) + D'*c(:,k-1) - Om*s(:,k-1);
  drdt     = -lam*r(:,k-1) + s(:,k-1);
  V(:,k)   = V(:,k-1) + dVdt*dt + sigV*randn(N,1).*sqrt(dt); 
  r(:,k)   = r(:,k-1) + drdt*dt;
  
  % knock-out neurons
  if t(k)>tko1, V(Nko1,k) = 0; end;
  if t(k)>tko2, V(Nko2,k) = 0; end;
  
  % check threshold crossings; only one neuron should spike per time step
  spiker  = find( V(:,k) > T);
  Nspiker = length(spiker);
  if Nspiker>0,
    chosen_to_spike=spiker(randi(Nspiker)); 
    s(chosen_to_spike,k)=1/dt;
  end

  % compute positive / negative inputs into example neurons
  for n=1:length(neuron)
    
    % excluding the self-reset
    allsynapticinputs = [D(:,neuron(n)) .* x(:,k-1); -Omrecurrent(neuron(n),:)'.*r(:,k-1)];
    Vpos(n,k) = sum( thlin( allsynapticinputs) );
    Vneg(n,k) = sum( -thlin( -allsynapticinputs) );

    % including the self-reset
    allinputs         = [D(:,neuron(n)) .* x(:,k-1); -Om(neuron(n),:)'.*r(:,k-1)];
    VRpos(n,k) = sum( thlin( allinputs) );
    VRneg(n,k) = sum( -thlin( -allinputs) );
  end
end

xest = D*r;                        % compute readout with original decoder

%% ========================= SIMULATION END ================================


%% ------------------------- PLOT RESULTS ----------------------------------

figure(3); clf;
set( gcf, 'Color', 'w' );
ncolor  = [0 0.7 0; 1 0.7 0; 0.6 0 0.8];     % color of highlighted neurons

%% ------------------------- FIG 3D, decoders ------------------------------
axscale = (round(100/N)+1)/100;
axes( 'pos', [0.1 0.85 0.16 0.12] );
hold on;
plot( D(1,:), D(2,:), 'k.', 'Markersize', 10 );
for k=1:length(neuron),
  pp = plot( D(1,neuron(k)), D(2,neuron(k)), '.', 'Markersize', 20 );
  set( pp, 'Color', ncolor(k,:) );
end
set( gca, 'XTick', [-0.04 0 0.04], 'YTick', [-0.04 0 0.04]);
box off;
xlabel( 'Decoder weight 1' ); 
ylabel( 'Decoder weight 2' );
axis([-1 1 -1 1]*axscale);

axes( 'pos', [0.4 0.85 0.16 0.12] );
hold on;
pp = plot( D(1,:), D(2,:), 'k.', 'Markersize', 10 );
set( pp, 'Color', [0.75 0.75 0.75]);
intact = setxor(1:N,Nko1);
plot( D(1,intact), D(2,intact), 'k.', 'Markersize', 10 );
for k=1:length(neuron),
  pp = plot( D(1,neuron(k)), D(2,neuron(k)), '.', 'Markersize', 20 );
  set( pp, 'Color', ncolor(k,:) );
end
set( gca, 'XTick', [-0.04 0 0.04], 'YTick', [-0.04 0 0.04]);
box off;
xlabel( 'Decoder weight 1' ); 
axis([-1 1 -1 1]*axscale);

axes( 'pos', [0.67 0.85 0.16 0.12] );
hold on;
pp = plot( D(1,:), D(2,:), 'k.', 'Markersize', 10 );
set( pp, 'Color', [0.75 0.75 0.75]);
intact = setxor(1:N,Nko2);
plot( D(1,intact), D(2,intact), 'k.', 'Markersize', 10 );
for k=1:length(neuron),
  pp = plot( D(1,neuron(k)), D(2,neuron(k)), '.', 'Markersize', 20 );
  set( pp, 'Color', ncolor(k,:) );
end
box off;
set( gca, 'XTick', [-0.04 0 0.04], 'YTick', [-0.04 0 0.04]);
xlabel( 'Decoder weight 1' ); 
axis([-1 1 -1 1]*axscale);

%% ------------------------- FIG 3E, input signal and estimate -------------
axes('pos',[0.1 0.66 0.8 0.12 ]);
Cest = [0.3 0.6 1];
Csig = [0 0 0];
hold on;
plot( t, xest(1,:)/2+0.5, 'Color', Cest, 'Linewidth', 3 );
plot( t, xest(2,:)/2-0.5, 'Color', Cest, 'Linewidth', 3 );
plot( t, x(1,:)/2+0.5, 'Color', Csig );
plot( t, x(2,:)/2-0.5, 'Color', Csig );
text( Timeon-30, 0.95, 'Signal x_1', 'Color', Csig, ...
'HorizontalAlignment','Left','VerticalAlignment','Middle');
text( Timeon-30, 0.65, 'Readout x_1', 'Color', Cest,...
'HorizontalAlignment','Left','VerticalAlignment','Middle');
text( Timeon-30, -0.5, 'Signal x_2', 'Color', Csig,...
'HorizontalAlignment','Left','VerticalAlignment','Middle');
text( Timeon-30, -0.8, 'Readout x_2', 'Color', Cest,...
'HorizontalAlignment','Left','VerticalAlignment','Middle');
set( gca, 'Visible', 'off' );
axis([Timeon Time -1.025 1.025 ]);

%% ------------------------- FIG 3F, raster plot ---------------------------
axes('pos',[0.1 0.39 0.8 0.25 ]);
hold on;
i    = find(s>0);                        % linear indices of all spikes
[yind,tind]   = ind2sub(size(s),i);      % matrix indices from linear indices
plot( t(tind),  mod(yind+N/2+1,N),  '.k', 'MarkerSize', 5);
for k=1:length(neuron)
  scop = s;
  scop(neuron(k),:)=-scop(neuron(k),:);
  isel = find(scop<0);                   % linear indices of neuron's spikes
  [ysind,tsind] = ind2sub(size(s),isel);
  plot( t(tsind), mod(ysind+N/2+1,N),  '.', 'MarkerSize', 12, ...
        'Color', ncolor(k,:));
end
axis([Timeon Time 0 N]);
axis off;
text( Timeon-30, N/2, 'Spike trains', 'Rotation', 90, ...
'HorizontalAlignment','Center','VerticalAlignment','Middle');
set( gca, 'TickDir', 'out');

%% ------------------------- FIG 3G, voltages of example neurons -----------
vpos = [0.31 0.255 0.2];
for k=1:length(neuron)
  axes('pos',[0.1 vpos(k) 0.8 0.05 ]);
  hold on;
  VT=T(neuron(k));                % voltage threshold of neuron
  thisV = V(neuron(k),:) + s(neuron(k),:)*0.8*max(max(V))*dt;
  plot(t,thisV,'-','LineWidth', 0.5, 'Color', ncolor(k,:));
  plot([0,Time],[VT,VT],'--k','LineWidth', 1);
  plot([0,Time],[-VT,-VT],'--k','LineWidth', 1);
  axis([Timeon Time -2.8*VT  2.8*VT]);
  %axis([Timeon Time -100*VT  3*VT]);
  set( gca,'XTick',[], 'XColor','w');
  set( gca,'YTick', [-VT 0 VT 2*VT] );
  set( gca,'YTickLabel', {'R', '', 'T', ''}, 'TickDir', 'out');
  if k==2, ylabel('Voltage (a.u.)'); end;
end

%% ------------------------- FIG 3H, E/(I+R) (where R = self-reset) --------
axes('pos',[0.1 0.11 0.8 0.075 ]);
hold on;
for k=1:length(neuron)
  plot(t,-VRpos(k,:)./VRneg(k,:),'-', 'LineWidth', 1.5, 'Color', ncolor(k,:) );
end
axis([Timeon Time 0 2.25]);
set( gca,'XTick',[], 'XColor','w');
set( gca,'YTick', [0 1 2] );
set( gca,'TickDir', 'out');
ylabel({'Total current', 'ratio [E/(I+R)]'});
% insert scale bar
pos = get(gca,'pos');
scl = (pos(3))/Time;
scalebar([0.9-scl*150, 0.02, scl*100, 0.015], '1 sec');

%% ------------------------- FIG 3I, E/I (no self-reset) -------------------
axes('pos',[0.1 0.007 0.8 0.075 ]);
hold on;
for k=1:length(neuron)
  plot(t,-Vpos(k,:)./Vneg(k,:),'-', 'LineWidth', 1.5, 'Color', ncolor(k,:) );
end
axis([Timeon Time 0 2.25]);
set( gca,'XTick',[], 'XColor','w');
set( gca,'YTick', [0 1 2] );
set( gca,'TickDir', 'out');
ylabel({'Syn. current', 'ratio [E/I]'});

%% ------------------------- KNOCK OUT LINES -------------------------------
axes('pos',[0 0 1 1]);
hold on;
xpos=0.365;
plot( [xpos xpos], [0.01 0.795], 'r--', 'Linewidth', 1);
plot( [xpos]*ones(1,4), 0.467:0.015:0.512, 'rx', ...
      'MarkerSize', 10, 'Linewidth', 2);
annotation('arrow', [xpos xpos], [0.81 0.78], 'Color', 'r', ...
           'Linewidth', 2 );
axis([0 1 0 1]);
set( gca, 'Visible', 'off' );
set( gca, 'Color', 'none' );

% plot knock-out line 2
axes('pos',[0 0 1 1]);
hold on;
xpos=0.635;
plot( [xpos xpos], [0.01 0.795], 'r--', 'Linewidth', 1);
plot( [xpos]*ones(1,4), 0.41:0.015:0.455, 'rx', ...
      'MarkerSize', 10, 'Linewidth', 2);
annotation('arrow', [xpos xpos], [0.81 0.78], 'Color', 'r', ...
           'Linewidth', 2 );
axis([0 1 0 1]);
set( gca, 'Visible', 'off' );
set( gca, 'Color', 'none' );

%--------------------------- PRINT RESULTS ---------------------------------

return; % no print by default

figureratio=1.2;
figurewidth=24;
cmtopix = 40;
gg = get( gcf, 'Position');
set( gcf, 'Position', [gg(1) gg(2) cmtopix*figurewidth cmtopix*figurewidth*figureratio]);
set( gcf, 'PaperUnits', 'centimeters');
set( gcf, 'PaperPosition', [0 0 figurewidth figurewidth*figureratio] );
print -depsc2 Figure3.eps
