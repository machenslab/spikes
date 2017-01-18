% This program produces Figure 3JK for "Optimal Compensation for
% Neuron Loss" by David Barrett, Sophie Deneve, and Christian Machens
%
% This program simulates a heterogenoues network of spiking neurons
% representing two sinusoidal signals.
% All free parameters can be changed from their default values 
% (within limits - note that plotting is not optimized for large changes)

clear all;
rng(12345);                      % initialize Random Number Generator
dosimulation = 1;                % set to zero if you only want to plot

%==========================  FIGURE 3JK ====================================

%--------------------------- FREE PARAMETERS -------------------------------

% Neurons
N = 32;                          % Number of Neurons in Network

% Refractory period in 10^{-2} sec
tref1 = 0;                       % during first simulation block
tref2 = 1.25;                    % during second simulation block
tref3 = 2;                       % during third simulation block

% Decoder
Dfixed(:,1:N) = [ sin( 2*pi*(1:N)/N ); cos( 2*pi*(1:N)/N )] / N;

% Other parameters
QBeta=0.05/N^2;                  % Quadratic cost
LBeta=0.15/N^2;                  % Linear cost
lam = 0.1;                       % lambda (1/lambda measured in 10^-2 sec)
sigV = 0.05/N^2;                 % standard deviation of voltage noise

% Number of repeated simulations per block (there are three blocks)
Nrep = 10;

%--------------------------- TIME PARAMETERS -------------------------------

Time= 1000;                      % Simulation time in 10^-2 sec
dt=0.01;                         % Euler time step in 10^-2 sec
t = 0:dt:Time-dt;                % array of time points
Nt=Time/dt;                      % Number of time steps
Timeon = 250;                    % Burn-in time (when plotting starts)
Ncycles = 4;                     % Number of sine wave sweeps;
Tcycle = Time/Ncycles;           % Time for each sweep;
tko1 = 2*Tcycle;                 % time of first knock-out
tko2 = 3*Tcycle;                 % time of second knock-out

%--------------------------- INPUT -----------------------------------------

f= Ncycles*2*pi/Time;            % Initialise Signal ...
x = [-sin(f*t); cos(f*t)];
dxdt = [ [0;0], diff(x')']/dt;   % Compute derivative
c = lam*x + dxdt;                % actual input into network


%=========================== SIMULATION START ==============================

if dosimulation,                 % three blocks with Nrep repeats

  refperiod = zeros(1,3*Nrep);   % refractory periods (in 10^-2 sec)
  refperiod(Nrep+1:2*Nrep)   = tref1;    % during first block
  refperiod(Nrep+1:2*Nrep)   = tref2;    % during second block
  refperiod(2*Nrep+1:3*Nrep) = tref3;    % during third block
  err = zeros(3*Nrep,N);         % average coding error
  
  for repeats=1:3*Nrep           % three blocks with Nrep repeats
    
    % Compute Connectivity
    D(:,1:N)=Dfixed;
    Om=D'*D + QBeta*eye(N);      % Initialize Recurrent connectivity
    T=(diag(Om)+QBeta+LBeta)/2;  % Thresholds
    
    % Loop over successive knock-outs of neurons
    for i=1:N
      
      Nleft = size(D,2)          % Number of neurons that are left
      
      % initial conditions
      V      = zeros(Nleft,Nt);
      s      = zeros(Nleft,Nt);
      r      = zeros(Nleft,Nt);
      eligible = ones(Nleft,1);  % non-refractory neurons
      
      % Actual network simulation
      for k=2:Nt
        
        % Voltage and firing rate update with Euler method
        dVdt     = -lam*V(:,k-1) + D'*c(:,k-1) - Om*s(:,k-1);
        drdt     = -lam*r(:,k-1) + s(:,k-1);
        V(:,k)   = V(:,k-1) + dVdt*dt +sigV*randn(Nleft,1).*sqrt(dt); 
        r(:,k)   = r(:,k-1) + drdt*dt;
        
        % check threshold crossings; only one neuron should spike
        % per time step; refractory neurons dont spike
        spiker  = find( V(:,k) > T & eligible>0 );
        Nspiker = length(spiker);
        if Nspiker>0,
          chosen_to_spike=spiker(randi(Nspiker)); 
          s(chosen_to_spike,k)=1/dt;
          eligible(chosen_to_spike) = - refperiod(repeats);
        end
        eligible = eligible + dt;
        
      end
      
      % Signal Estimate and Error
      xest = D*r;
      err(repeats,i) = norm( x(t>Timeon) - xest(t>Timeon)) / norm( x(t>Timeon) );
      
      % Eliminate a random neuron and shrink the network
      tobekilled = randi(Nleft);
      intact = setxor(1:Nleft,tobekilled); 
      D = D(:,intact);
      Om = Om(:,intact);
      Om = Om(intact,:);
      T = T(intact);
    end %loop over successive knock-outs
  end % loop over repeats

  save Figure3JK_sim.mat err;

else
  
  load Figure3JK_sim.mat

end

% Redivide errors into the three blocks with different refractory periods
err0 = 100*[err(1:Nrep,:), ones(Nrep,1)];
err1 = 100*[err(Nrep+1:2*Nrep,:), ones(Nrep,1)];
err2 = 100*[err(2*Nrep+1:3*Nrep,:), ones(Nrep,1)];
pcko = 100*(0:N)/N; % percentage of k.o. neurons

%% ========================= SIMULATION END ================================


%% --------------------------- PLOT RESULTS ----------------------------------

figure(33); clf;
set( gcf, 'Color', 'w' );

subplot(1,2,1); hold on;
for k=1:Nrep
  pp = plot( pcko, err0(k,:), 'k.-' );
  set( pp, 'Color', [0.7 0.7 0.7] );
end
pp = plot( pcko, mean(err0), 'k.-', 'Linewidth', 2 );
axis( [0 100 0 100] );
xlabel('% Cell death' );
ylabel('% Error' );

subplot(1,2,2); hold on;
for k=1:Nrep
  pp = plot( pcko, err1(k,:), 'k.-' );
  set( pp, 'Color', [0.7 0.7 0.7] );
end
pp = plot( pcko, mean(err1), 'k.-', 'Linewidth', 2 );
axis( [0 100 0 100] );
xlabel('% Cell death' );
ylabel('% Error' );

%--------------------------- PRINT RESULTS ---------------------------------

return; % no print by default

figureratio=0.2;
figurewidth=24;
cmtopix = 24;
gg = get( gcf, 'Position');
set( gcf, 'Position', [gg(1) gg(2) cmtopix*figurewidth cmtopix*figurewidth*figureratio]);
set( gcf, 'PaperUnits', 'centimeters');
set( gcf, 'PaperPosition', [0 0 figurewidth figurewidth*figureratio] );
print -depsc2 Figure3JK.eps
