% This program produces Figure 5 for "Optimal Compensation for
% Neuron Loss" by David Barrett, Sophie Deneve, and Christian Machens
%
% This program computes the tuning curves for three example
% networks before and after neuron loss, using quadratic programming
% All free parameters can be changed from their default values 
% (within limits - note that plotting is not optimized for large changes)

clear all;
rng('default');

%=========================== FIGURE 5 A-D ===================================

%--------------------------- FREE PARAMETERS --------------------------------

% Neurons
N = 16;                          % Number of Neurons in Network
                                 % (should be an even number)
Nko1 = 13:16;                    % Indices of neurons in first k.o.
Nko2 = 9:16;                     % Indices of neurons in second k.o.

% Decoder
Gam     = 4*(0:N/2-1)/(N/2-1) + ones(1,N/2);
G       = [Gam, -Gam; 0.5*ones(1,N)]/(100*N);
G = G.*(rand(2,N)+0.15);         % read-out weights

% Decoder adjustment for better visibility (non-essential, used for paper)
% G(1,10) = G(1,10)*0.7;
% G(2,10) = G(2,10)*1.5;
% G(1,11) = G(1,11)*0.8;

% Other parameters
lam = 0.001;                     % Decoder timescale (in inverse milliseconds)
QBeta=0.0004/N^2;                % Quadratic firing rate cost

%--------------------------- STIMULI / SIGNALS ------------------------------

Nx=30;                           % Number of x-axis samples
X=(0:Nx-1)/(Nx-1)*2-1;           % Calculate x-axis values

%--------------------------- KNOCK OUTS ------------------------------------- 

D{1} = G;                        % Decoder for full network
D{2} = G; D{2}(:,Nko1) = 0;      % Decoder after first k.o.
D{3} = G; D{3}(:,Nko2) = 0;      % Decoder after second k.o.

%--------------------------- TUNING CURVES AND K.O. VIA QP ------------------

Rp = zeros( 3, Nx, N );          % predicted firing rates

% Loop over k.o. schedules
for j=1:3
  W=D{j}'*D{j} + QBeta*eye(N);   % Recurrent connectivity

  % Loop over stimulus values
  for i=1:Nx
    % signal value in this iteration
    x=[X(i) 0.8/sqrt(N)]';       

    % estimate firing rates using quadratic programming by reformulating
    % the problem as a linear least square minimization with
    % positivity constraint (this way we dont need the optimization toolbox)
    CC = sqrtm( W );
    d = CC \ (D{j}'*x);
    Rp(j,i,:) = lsqnonneg( CC, d);
  end
end
rate = Rp*lam*1000;              % firing rate expressed in Hz

% compute signal estimates for scenarios with and w/o compensation
xest_or   = D{1}(1,:) * squeeze( Rp(1,:,:))';     % original
xest_cp   = D{2}(1,:) * squeeze( Rp(2,:,:))';     % first k.o. with compensation
xest_cp2  = D{3}(1,:) * squeeze( Rp(3,:,:))';     % second k.o. with compensation
xest_nocp = D{2}(1,:) * squeeze( Rp(1,:,:))';     % first k.o. w/o compensation
xest_nocp2= D{3}(1,:) * squeeze( Rp(1,:,:))';     % second k.o. w/o compensation

% compute errors for scenarios with and w/o compensation
Err_or    = mean( (X-xest_or)'*(X-xest_or) );     % original
Err_cp    = mean( (X-xest_cp)'*(X-xest_cp) );     % first k.o. with compensation
Err_cp2   = mean( (X-xest_cp2)'*(X-xest_cp2) );   % second k.o. with compensation
Err_nocp  = mean( (X-xest_nocp)'*(X-xest_nocp) ); % first k.o. w/o compensation
Err_nocp2 = mean( (X-xest_nocp2)'*(X-xest_nocp2) );% second k.o. w/o compensation

%--------------------------- PLOT RESULTS ----------------------------------

figure(5); clf;
set( gcf, 'Color', 'w' );

% Design rainbow color map for different tuning curves
cmap   = colormap('jet');
Nc     = size(cmap, 1);
cl     = round( (1:N) * Nc/N );
colour = cmap(Nc-cl+1,:);
colour(N/2+1:N,:) = colour(N:-1:N/2+1,:);

% Subplot spacing
h = [0.09 0.24 0.4 0.54 0.7 0.84];
v = [0.7 0.4 0.1];
w = 0.1;
l = 0.2;

% Axis parameters
xmarks = [-1, -0.5, 0, 0.5, 1];
xlabels= {'-1', '', '0', '', '1'};
xleft  = -1; 
xright = 1;

% Panel A: Decoding Weights
subplot('Position', [h(1) v(1) w l]);
for i=N:-1:1
  plot([D{1}(1,i)],[D{1}(2,i)],'.', 'MarkerSize', 25, 'color',colour(i,:));
  hold on;
end
axis([-0.004 0.004 -0.001 0.001]);
set(gca,'XTick',[-0.004, 0, 0.004], 'XTickLabel', {'-.004', '0', '.004'} );
set(gca,'YTick',[-0.001, 0, 0.001], 'YTickLabel', {'-.001', '0', '.001'} );
set(gca,'TickDir', 'out', 'TickLength', [0.02 0.02] );
box off;
ylabel('Decoder weight 2');
xlabel('Decoder weight 1');

% Panel B: Tuning Curves
subplot('Position', [h(2) v(1) w l]);
for i=N:-1:1
  plot( X, rate(1,:,i), '-', 'LineWidth', 2, 'color', colour(i,:) );
  hold on;
end
axis([xleft xright 0 120]);
box off;
set(gca,'XTick',xmarks,'XTickLabel',xlabels, 'YTick', [0 50 100]);
set(gca,'TickDir', 'out', 'TickLength', [0.02 0.02] );
ylabel('Firing rate (Hz)');
xlabel('Signal x');

% Panel C: Tuning Curves
subplot('Position', [h(3) v(1) w l]);
for i=N:-1:1
  plot( X, rate(2,:,i), '-', 'LineWidth', 2, 'color', colour(i,:) );
  hold on;
end
axis([xleft xright 0 120]);
box off;
set(gca,'XTick',xmarks,'XTickLabel',xlabels, 'YTick', [0 50 100]);
set(gca,'TickDir', 'out', 'TickLength', [0.02 0.02] );
ylabel('Firing rate (Hz)');

% Panel D: Tuning Curves
subplot('Position', [h(5) v(1) w l]);
for i=N:-1:1
  plot( X, rate(3,:,i), '-', 'LineWidth', 2, 'color', colour(i,:) );
  hold on;
end
axis([xleft xright 0 120]);
box off;
set(gca,'XTick',xmarks,'XTickLabel',xlabels, 'YTick', [0 50 100]);
set(gca,'TickDir', 'out', 'TickLength', [0.02 0.02] );
ylabel('Firing rate (Hz)');

% Panel C: Errors
subplot('Position', [h(4) v(1) w l]);
hold on;
plot( X, Err_cp, 'k-', 'Linewidth', 2 );
plot( X, Err_nocp, 'k--', 'Linewidth', 2, 'Color', [0.65 0.65 0.65]);
axis([xleft xright -0.03 0.3]);
set(gca,'XTick',xmarks,'XTickLabel',xlabels,'Ytick', [0 0.1 0.2]);
set(gca,'TickDir', 'out', 'TickLength', [0.02 0.02] );
xlabel('Signal x');
ylabel('Error E');
box off;

% Panel D: Errors
subplot('Position', [h(6) v(1) w l]);
hold on;
plot( X, Err_cp2, 'k-', 'Linewidth', 2 );
plot( X, Err_nocp2, 'k--', 'Linewidth', 2, 'Color', [0.65 0.65 0.65]);
axis([xleft xright -0.03 0.3]);
set(gca,'XTick',xmarks,'XTickLabel',xlabels,'Ytick', [0 0.1 0.2]);
set(gca,'TickDir', 'out', 'TickLength', [0.02 0.02] );
xlabel('Signal x');
ylabel('Error E');
box off;



%=========================== FIGURE 5 I-K ===================================

%--------------------------- FREE PARAMETERS --------------------------------

% Neurons
N = 4;                           % Number of Neurons in Network
Nko = 1;                         % Neuron to be knocked out

% Decoder Parameters
Amp = 5/N;                       % read-out weight amplitudes
angles = (0:N-1)*(2*pi/N);       % read-out weight angles

% Other parameters
lam = 0.1;                       % Decoder timescale (in inverse milliseconds)
QBeta=0.1/N^2;                   % Quadratic firing rate cost

%--------------------------- STIMULI / SIGNALS ------------------------------

Nx=60;
theta_all=(0:Nx-1)/(Nx-1)*2*pi-pi;
angdeg   = 360*theta_all'/(2*pi); % in degrees
x = [ cos(theta_all') sin(theta_all') ]';

%--------------------------- KNOCK OUTS ------------------------------------- 

D{1}   = Amp * [cos(angles); sin(angles) ]; % read-out weights
D{2}   = D{1};
D{2}(:,Nko) = [ 0 0 ];             % kill one neuron

%--------------------------- TUNING CURVES AND K.O. VIA QP ------------------

Rp = zeros( 2,Nx, N);            % predicted firing rates

% Loop over k.o. schedules
for j=1:2
  W=D{j}'*D{j} +QBeta*eye(N);    % recurrent connectivity
  for i=1:Nx
    d  = sqrtm( inv(W) ) * D{j}' * x(:,i);
    Rp(j,i,:) = lsqnonneg( sqrtm(W), d ); % quadratic programming
  end
end
rate = Rp*lam*1000;              % rates in Hz

% compute signal estimates for scenarios with and w/o compensation
xest_or  = D{1} * squeeze( Rp(1,:,:))';      % original
xest_cp  = D{2}   * squeeze( Rp(2,:,:))';    % k.o. with compensation
xest_nocp= D{2} * squeeze( Rp(1,:,:))';      % k.o. without compensation

% compute errors for scenarios with and w/o compensation
Err_or   = mean( (x-xest_or)'*(x-xest_or) ); % original
Err_cp   = mean( (x-xest_cp)'*(x-xest_cp) ); % k.o. with compensation
Err_nocp = mean( (x-xest_nocp)'*(x-xest_nocp) ); % k.o. w/o compensation

%--------------------------- PLOT RESULTS ----------------------------------

% Design color map for different tuning curves
cmap   = colormap('summer');
Nc     = size(cmap, 1);
cl     = round( (1:N) * Nc/N );
colour = cmap(Nc-cl+1,:);

% Axis parameters
xmarks = [-180:90:180];
xlabels= {'-180', '', '0', '', '180'};
xleft  = -180; 
xright = 180;

% Panel I: Decoding Weights
subplot('Position', [h(1) v(3) w l]);
for i=N:-1:1
  plot([D{1}(1,i)],[D{1}(2,i)],'.', 'MarkerSize', 25, 'color',colour(i,:));
  hold on;
end
axis([-1.5 1.5 -1.5 1.5]);
set(gca,'XTick',-1:0.5:1, 'XTickLabel',{'-1','','0','','1'});
set(gca,'YTick',-1:0.5:1, 'YTickLabel',{'-1','','0','','1'});
set(gca,'TickDir', 'out', 'TickLength', [0.02 0.02] );
box off;
ylabel('Decoder weight 2');
xlabel('Decoder weight 1');

% Panel J: Tuning Curves
subplot('Position', [h(2) v(3) w l]);
for i=N:-1:1
  plot( angdeg, rate(1,:,i), '-', 'LineWidth', 2, 'color', colour(i,:) );
  hold on;
end
axis([xleft xright 0 100]);
box off;
set(gca,'XTick',xmarks,'XTickLabel',xlabels);
set(gca,'TickDir', 'out', 'TickLength', [0.02 0.02] );
ylabel('Firing rate (Hz)');
xlabel('Direction \theta (deg)');

% Panel K: Tuning Curves
subplot('Position', [h(5) v(3) w l]);
for i=1:N
  plot( angdeg, rate(2,:,i), '-', 'LineWidth', 2, 'color', colour(i,:) );
  hold on;
end
axis([xleft xright 0 100]);
box off;
set(gca,'XTick',xmarks,'XTickLabel',xlabels);
set(gca,'TickDir', 'out', 'TickLength', [0.02 0.02] );
ylabel('Firing rate (Hz)');
xlabel('Direction \theta');

% Panel K: Errors
subplot('Position', [h(6) v(3) w l]);
hold on;
plot( angdeg, Err_cp, 'k-', 'Linewidth', 2 );
plot( angdeg, Err_nocp, 'k--', 'Linewidth', 2, 'Color', [0.65 0.65 0.65]);
axis([xleft xright -0.03 0.4]);
set(gca,'XTick',xmarks,'XTickLabel',xlabels,'Ytick', [0 0.1 0.2 0.3]);
set(gca,'TickDir', 'out', 'TickLength', [0.02 0.02] );
xlabel('Direction \theta (deg)');
ylabel('Error E');
box off;




%=========================== FIGURE 5 E-H ===================================


%--------------------------- FREE PARAMETERS --------------------------------

% Neurons
N = 20;                          % Number of Neurons in Network
Nko1 = 1:5;                      % Indices of neurons in first k.o.
Nko2 = [1:7,18:20];              % Indices of neurons in second k.o.

% Decoder Parameters
Amp = 5/N;                       % read-out weight amplitudes
Amprand = 2.5/N;                 % random amplitude component
angles = -pi/4 + 2*pi*(1:N)/N;   % read-out weight angles

% Other parameters
lam = 0.1;                       % Decoder timescale (in inverse milliseconds)
QBeta=0.1/N^2;                   % Quadratic firing rate cost 

%--------------------------- STIMULI / SIGNALS ------------------------------

Nx=60;
theta_all=(0:Nx-1)/(Nx-1)*2*pi-pi;
angdeg   = 360*theta_all'/(2*pi); % in degrees
x = [ cos(theta_all') sin(theta_all') ]';

%--------------------------- KNOCK OUTS ------------------------------------- 

Grand  = Amp*ones(1,N) + Amprand*rand(1,N);
G      = [Grand.*cos(angles); Grand.*sin(angles)];
D{1} = G;                        % Decoder for full network
D{2} = G; D{2}(:,Nko1) = 0;      % Decoder after first k.o.
D{3} = G; D{3}(:,Nko2) = 0;      % Decoder after second k.o.

%--------------------------- TUNING CURVES AND K.O. VIA QP ------------------

Rp = zeros( 3, Nx, N);           % predicted firing rates

% Loop over k.o. schedules
for j=1:3
  W=D{j}'*D{j} +QBeta*eye(N);    % recurrent connectivity
  for i=1:Nx
    d  = sqrtm( inv(W) ) * D{j}' * x(:,i);
    Rp(j,i,:) = lsqnonneg( sqrtm(W), d ); % quadratic programming
  end
end
rate = Rp*lam*1000;              % rates in Hz

% compute signal estimates for scenarios with and w/o compensation
xest_or   = D{1} * squeeze( Rp(1,:,:))';     % original
xest_cp   = D{2}   * squeeze( Rp(2,:,:))';   % first k.o. with compensation
xest_cp2  = D{3} * squeeze( Rp(3,:,:))';     % second k.o. with compensation
xest_nocp = D{2} * squeeze( Rp(1,:,:))';     % first k.o. w/o compensation
xest_nocp2= D{3} * squeeze( Rp(1,:,:))';     % second k.o. w/o compensation

% compute errors for scenarios with and w/o compensation
Err_or  = mean( (x-xest_or)'*(x-xest_or) );  % original
Err_cp  = mean( (x-xest_cp)'*(x-xest_cp) );  % first k.o. with compensation
Err_cp2  = mean( (x-xest_cp2)'*(x-xest_cp2) );% second k.o. with compensation
Err_nocp= mean( (x-xest_nocp)'*(x-xest_nocp) );% first k.o. w/o compensation
Err_nocp2= mean( (x-xest_nocp2)'*(x-xest_nocp2) );% second k.o. w/o compensation

%--------------------------- PLOT RESULTS ----------------------------------

% Design color map for different tuning curves
cmap   = colormap('winter');
Nc     = size(cmap, 1);
cl     = round( (1:N) * Nc/N );
colour = cmap(Nc-cl+1,:);

% Panel E: Decoding Weights
subplot('Position', [h(1) v(2) w l]);
for i=1:N
  plot([D{1}(1,i)],[D{1}(2,i)],'.', 'MarkerSize', 25, 'color',colour(i,:));
  hold on;
end
axis([-0.5 0.5 -0.5 0.5]);
set(gca,'XTick',-0.5:0.5:0.5);
set(gca,'YTick',-0.5:0.5:0.5);
set(gca,'TickDir', 'out', 'TickLength', [0.02 0.02] );
box off;
ylabel('Decoder weight 2');
xlabel('Decoder weight 1');

% Panel F: Tuning Curves (Intact system)
subplot('Position', [h(2) v(2) w l]);
for i=N:-1:1
  plot( angdeg, rate(1,:,i), '-', 'LineWidth', 2, 'color', colour(i,:) );
  hold on;
end
axis([xleft xright 0 180]);
box off;
set(gca,'XTick',xmarks,'XTickLabel',xlabels);
set(gca,'YTick',[0:50:150] );
set(gca,'TickDir', 'out', 'TickLength', [0.02 0.02] );
ylabel('Firing rate (Hz)');
xlabel('Direction \theta (deg)');

% Panel G: Tuning Curves (First k.o.)
g = subplot('Position', [h(3) v(2) w l*1.25]);
for i=N:-1:1
  plot( angdeg, rate(2,:,i), '-', 'LineWidth', 2, 'color', colour(i,:) );
  hold on;
end
plot([xleft, xleft], [180,225], 'w', 'Linewidth', 10 );
axis([xleft xright 0 225]);
box off;
set(gca,'XTick',xmarks,'XTickLabel',xlabels);
set(gca,'YTick',[0:50:150] );
set(gca,'TickDir', 'out', 'TickLength', [0.02 0.02] );
ylabel('Firing rate (Hz)');
xlabel('Direction \theta (deg)');

% Panel G: Error
subplot('Position', [h(4) v(2) w l]);
hold on;
plot( angdeg, Err_cp, 'k-', 'Linewidth', 2 );
plot( angdeg, Err_nocp, 'k--', 'Linewidth', 2, 'Color', [0.65 0.65 0.65]);
axis([xleft xright -0.03 0.4]);
set(gca,'XTick',xmarks,'XTickLabel',xlabels);
set(gca,'TickDir', 'out', 'TickLength', [0.02 0.02],'YTick',[0 0.1 ...
                    0.2 0.3] );
xlabel('Direction \theta (deg)');
ylabel('Error E');
box off;

% Panel H: Tuning Curves (Second k.o.)
g = subplot('Position', [h(5) v(2) w l*1.25]);
for i=N:-1:1
  plot( angdeg, rate(3,:,i), '-', 'LineWidth', 2, 'color', colour(i,:) );
  hold on;
end
plot([xleft, xleft], [180,225], 'w', 'Linewidth', 10 );
axis([xleft xright 0 225]);
box off;
set(gca,'XTick',xmarks,'XTickLabel',xlabels);
set(gca,'YTick',[0:50:150] );
set(gca,'TickDir', 'out', 'TickLength', [0.02 0.02] );
ylabel('Firing rate (Hz)');
xlabel('Direction \theta (deg)');

% Panel H: Error
subplot('Position', [h(6) v(2) w l]);
hold on;
plot( angdeg, Err_cp2, 'k-', 'Linewidth', 2 );
plot( angdeg, Err_nocp2, 'k--', 'Linewidth', 2, 'Color', [0.65 0.65 0.65]);
axis([xleft xright -0.03 0.4]);
set(gca,'XTick',xmarks,'XTickLabel',xlabels,'YTick',[0 0.1 0.2 0.3]);
set(gca,'TickDir', 'out', 'TickLength', [0.02 0.02] );
xlabel('Direction \theta (deg)');
ylabel('Error E');
box off;

%--------------------------- PRINT RESULTS ---------------------------------

return; % by default dont print
figureratio=0.5;
figurewidth=40;
cmtopix = 24;
gg = get( gcf, 'Position');
set( gcf, 'Position', [gg(1) gg(2) cmtopix*figurewidth cmtopix*figurewidth*figureratio]);
set( gcf, 'PaperUnits', 'centimeters');
set( gcf, 'PaperPosition', [0 0 figurewidth figurewidth*figureratio] );
print -depsc2 Figure5.eps
















