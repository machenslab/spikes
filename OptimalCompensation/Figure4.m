% This program produces Figure 4 for "Optimal Compensation for
% Neuron Loss" by David Barrett, Sophie Deneve, and Christian Machens
%
% This program compares the tuning curves of three simulated
% spiking networks with the tuning curves predicted by quadratic
% programming.
%
% All free parameters can be changed from their default values 
% (within limits - note that plotting is not optimized for large changes)

clear all;
rng('default');                  % initialize Random Number Generator
 
%==========================  FIGURE 4 =======================================

%--------------------------- FREE PARAMETERS -------------------------------

% Neurons
N{1} = 2;                        % Number of Neurons in Fig. 4a
N{2} = 16;                       % Number of Neurons in Fig. 4b
N{3} = 16;                       % Number of Neurons in Fig. 4c

% Decoder
D{1} = [2 -2; 0.1 0.1]/80;       % read-out weights: Fig. 4a
Dam   =  4*( 0:N{2}/2-1 ) / (N{2}/2-1) + ones( 1, N{2}/2 );
D{2}  = [Dam, -Dam; 0.5*(ones(1,N{2}))]/(100*N{2}); % Fig. 4b
D{3} = D{2}.*(rand(2,N{3})+0.15);% read-out weights: Fig. 4c

% Other parameters
QBeta = 0.0004;                  % Quadratic Cost (will be divided by N)
lam   = 0.001;                   % Decoder timescale (in inverse milliseconds)
sigV  = 0.00000005;              % voltage noise (will be divided by N^2)

% Simulation length (In paper, we use Time=10000 and TimeStart=2000)
Time=1000;                       % Simulation time in miliseconds
TimeStart=500;                   % Start Time for Firing rate calculations

%--------------------------- TIME PARAMETERS --------------------------------

dt=0.01;                         % Euler time step in milliseconds
t = 0:dt:Time-dt;                % array of time points
Nt=length(t);                    % Number of time steps

%--------------------------- INPUT SIGNALS ----------------------------------

Nx=30;                           % Number of x-axis samples
X = (0:Nx-1)/(Nx-1)*2 - 1;       % Calculate x-axis values


%=========================== SIMULATION START ==============================

for j=1:3
  
  Om=D{j}'*D{j} + QBeta/N{j}^2*eye(N{j}); % Recurrent connectivity
  T=diag(Om)/2;                  % Thresholds

  for i=1:Nx                     % loop over input signals
    
    % signal values in this iteration
    x=[X(i) 0.8/sqrt(N{j})]';

    % estimate firing rates using quadratic programming by reformulating
    % the problem as a linear least square minimization with
    % positivity constraint (this way we dont need the optimization toolbox)
    CC = sqrtm( Om );
    d = CC \ (D{j}'*x);
    rini = lsqnonneg( CC, d);

    % simulate spiking network
    c = lam*x*ones(1,Nt) + x*[1/dt, zeros(1,Nt-1)];  % constant (step) input
    sigV2 = sigV/N{j}^2;
    
    % initialize variables
    V = zeros( N{j}, Nt);        % membrane voltages
    s = zeros( N{j}, Nt);        % spike trains
    r = zeros( N{j}, Nt);        % firing rates
    
    % simulate network
    for k=2:Nt
      
      % Voltage and firing rate update with Euler method
      dVdt     = -lam*V(:,k-1) + D{j}'*c(:,k-1) - Om*s(:,k-1);
      drdt     = -lam*r(:,k-1) + s(:,k-1);
      V(:,k)   = V(:,k-1) + dVdt*dt + sigV2*randn(N{j},1)*sqrt(dt); 
      r(:,k)   = r(:,k-1) + drdt*dt;
      
      % check threshold crossings, only one neuron should spike per time step
      spiker  = find( V(:,k) > T);
      Nspiker = length(spiker);
      if Nspiker>0,
        chosen_to_spike=spiker(randi(Nspiker)); 
        s(chosen_to_spike,k)=1/dt;
      end
      
    end
        
    % store predicted and measured firing rates
    % ( wait for firing rates to set to a stationary state, hence
    % we start computation at TimeStart )
    Rp{j}(i,:)    = rini;                           % predicted firing rate
    Rm{j}(i,:)    = mean(r(:,TimeStart/dt+1:Nt)');  % measured firing rate
  end  
end

%--------------------------- PLOT RESULTS --------------------------------

figure(4); clf;
set( gcf, 'Color', 'w' );

col = rbcolors(3*N{3}/2)*0.85;
color{1} = col( [5*N{3}/4+1, N{3}/4]-1, : );
color{2} = col( [N{3}+1:3*N{3}/2, N{3}/2:-1:1],:);
color{3} = col( [N{3}+1:3*N{3}/2, N{3}/2:-1:1],:);
yax = [100, 100, 100];

for j=1:3
        
  % plot decoder weights
  subplot(3,4,(j-1)*4+1);
  hold on;
  for i=1:N{j}
    plot(D{j}(1,i),D{j}(2,i)/0.02,'.','MarkerSize', 16, 'color',color{j}(i,:));
  end
  if j==3
    xlabel('Decoder weight 1');
  end
  ylabel('Decoder weight 2');
  if j==1
    axis([-0.05 0.05 0 0.125]); 
    set( gca, 'XTick', -0.05:0.05/2:0.05,'XTickLabel', {'-0.05','','0','','0.05',''} );
    set( gca, 'YTick', 0:0.125/4:0.125,'YTickLabel', {'0','','', ...
                        '','0.125'} );    
  else
    axis([-0.005 0.005 0 0.025]); 
    set( gca, 'XTick', -0.005:0.005/2:0.005,'XTickLabel', {'-0.005','','0','','0.005',''} );
    set( gca, 'YTick', 0:0.025/4:0.025,'YTickLabel', {'0','','', ...
                        '','0.025'} );
  end
  set( gca, 'TickDir', 'out','TickLength', [0.035,0.035] );
  box off;

  % plot predicted tuning curves
  subplot(3,4,(j-1)*4+2); 
  hold on;
  for i=1:N{j}
    plot(X',Rp{j}(:,i),'-','LineWidth', 1.5,'color',color{j}(i,:));
  end
  if j==1
    title('Predicted');
    annotation('arrow', [0.442 0.442], [0.86 0.81], 'Color', 'k', ...
               'Linewidth', 2 );
  end
  if j==3
    xlabel('Signal x');
  end
  ylabel('Firing rate (Hz)');
  axis([-1.05 1 0 yax(j)]);
  set(gca,'XTick', -1:0.5:1, 'YTick',0:25:100)
  set(gca,'YTickLabel',{'0','','50','','100'});
  if j==1
      axis([-1.05 1 0 40]);
      set(gca,'YTick', 0:10:40,'YTickLabel',{'0','','20','','40'});
  end
  set( gca,'LineWidth',1, 'TickDir', 'out','TickLength', [0.035,0.035] );
  box off;
  
  % plot simulated tuning curves  
  subplot(3,4,(j-1)*4+3);
  hold on;
  for i=1:N{j}
    plot(X',Rm{j}(:,i),'-','LineWidth', 1.5,'color',color{j}(i,:));
  end
  if j==1
    title('Measured');
  elseif j==3
    xlabel('Signal x');
  end
  ylabel('Firing rate (Hz)');
  axis([-1.05 1 0 yax(j)]);
  set(gca,'XTick', -1:0.5:1, 'YTick',0:25:100)
  set(gca,'YTickLabel',{'0','','50','','100'});
  set( gca,'LineWidth',1, 'TickDir', 'out','TickLength', [0.035,0.035] );
    if j==1
      axis([-1.05 1 0 40]);
      set(gca,'YTick', 0:10:40,'YTickLabel',{'0','','20','','40'});
  end
  box off;
  
  % compare predicted and simulated tuning curves
  subplot(3,4,(j-1)*4+4)
  plot([0 yax(j)],[0 yax(j)],'--r','LineWidth', 1.5);
  hold on;
  plot(reshape(Rm{j}(:,:),Nx*N{j},1) ,reshape(Rp{j},Nx*N{j},1),'.k','MarkerSize', 10)
  if j==3
    xlabel('Measured (Hz)');
  end
  ylabel('Predicted (Hz)');
  set(gca,'XTick',0:25:100, 'YTick',0:25:100)
  set(gca,'YTickLabel',{'0','','50','','100'});
  set(gca,'XTickLabel',{'0','','50','','100'});
  set( gca,'LineWidth',1, 'TickDir', 'out','TickLength', [0.035,0.035] );
  axis([0 yax(j) 0 yax(j)]);
  if j==1
      axis([0 40 0 40]);
      set(gca,'XTick',0:10:40, 'YTick',0:10:40)
      set(gca,'YTickLabel',{'0','','20','','40'});
      set(gca,'XTickLabel',{'0','','20','','40'});
  end
  box off;
  
end
