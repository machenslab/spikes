% This program produces Figure 6 for "Optimal Compensation for
% Neuron Loss" by David Barrett, Sophie Deneve, and Christian Machens
%
% This program simulates tuning curves in an oculomotor integrator
% both before and after neuron loss, effect of neuron loss is plotted.
%
% All free parameters can Ebe changed from their default values 
% (within limits - note that plotting is not optimized for large changes)

clear all;
rng('default');                  % initialize Random Number Generator

%==========================  FIGURE 6 BD ===================================

%--------------------------- FREE PARAMETERS -------------------------------

% Neurons
N = 40;                          % Number of Neurons in Network (even!)
Nko = N/2+1:N;                   % knock-out of left half
Nf = 15;                         % Number of fish to sample from

% Decoder
Gam = 0.01*ones(1,N/2) + 0.04*(0:N/2-1)/(N/2-1);
BTV = 0.02/sqrt(N);                % Background task variable
G = [Gam,-Gam; ones(1,N)*BTV ]/N;  % both positive + negative read-out weights

% Other parameters
lam = 0.001;                     % Decoder timescale (in inverse milliseconds)
Mean_Firing_Rate_Hz=50;          % in (Hz) Target mean population firing rate
QBeta=0.0004/N^2;                % Quadratic Cost

%--------------------------- INPUT SIGNALS ----------------------------------

Nx=31;                           % Number of x-axis samples
X_max=1;                         % Maximum x-axis value
X_min=-1;                        % Minimum x-axis value
nu0=Mean_Firing_Rate_Hz/1000/lam;
X=(0:Nx-1)/(Nx-1)*(X_max-X_min)+X_min;

%=========================== SIMULATION START ===============================

%--------------------------- CALCULATE TUNING CURVES ------------------------

k=0;
Rp = zeros( Nx, N );             % Tuning Curves for one goldfish
Rp1 = zeros( Nx, 2*Nf );         % Sample of tuning curves across fish
Greg=G;

for j=1:Nf 

    % add randomness to decoder with multiplicative noise
    G      = Greg.*(rand(2,N)+0.5); 
    [Y, I] = sort( G(1,:),'descend' );
    G      = G(:,I)*(lam*1000);

    F=G';                              % Feedforward connectivity
    W=G'*G +QBeta*eye(N);               % Recurrent connectivity

    % Loop over stimulus values
    for i=1:Nx
      x=[X(i) nu0*BTV]';
      
      % estimate firing rates using quadratic programming by reformulating
      % the problem as a linear least square minimization with
      % positivity constraint (this way we dont need the optimization toolbox)
      CC = sqrtm( W );
      d  = CC \ (F*x);
      Rp(i,:) = lam*1000*lsqnonneg(CC,d);             
    end
    
    % randomly pick two tuning curves from one side
    Rp1(:,k+1)=Rp(:,randi(N/2));
    Rp1(:,k+2)=Rp(:,randi(N/2));
    k=k+2;

end

%--------------------------- FIT THRESHOLD-LINEARITY ----------------------

% Tuning curve processing that matches tuning curve processing for
% Aksay et al 2000 Fig. 5 (D) - Threshold-linear fit

% Preprocessing
for i=1:Nx
  for j=1:2*Nf
    if Rp1(i,j)<5
      Rp1(i,j)=0;
    end
  end
end

% Fit threshold-linear function
for j=1:2*Nf
  
  % Extract linear part (above threshold)
  ind = find( Rp1(:,j)>0 );
  R4 = Rp1( ind, j )';
  X4 = X( ind );
  
  % Fit linear part
  p(j,:) = polyfit(X4,R4,1);
  kN(j)=p(j,1);                  % slope
  Eth(j)= -p(j,2)/p(j,1);        % Intersect
  ymax(j) = p(j,1)*1 + p(j,2);
  
  % Compute threshold-linear curve
  Rp2(j,:)=[5 ymax(j)];
  X2(j,:)=[(5-p(j,2))/p(j,1) 1 ];
  
end

%--------------------------- LEFT-SIDE KNOCKOUT ------------------------

% use only eight x-axis values, as in experiments
Nx=8;
X10=(0:Nx-1)/(Nx-1)*(X_max-X_min)+X_min; 

F=G';                            % Feedforward connectivity
W=G'*G +QBeta*eye(N);             % Recurrent connectivity
CC = sqrtm( W );

% Compute tuning curves before neuron loss (same as above)
for i=1:Nx
  
  x=[X10(i) nu0*BTV]';
  d  = CC \ (F*x);
  rp = lsqnonneg(CC,d);             
  
  % estimate representation error
  xestp=G*rp;
  Ebefore(i)=mean((x - xestp).*(x - xestp))+QBeta*(rp'*rp);
  
end

% Knockout
Gko=G;
Gko(:,Nko)=0;

F=Gko';                          % Feedforward connectivity
W=Gko'*Gko +QBeta*eye(N);         % Recurrent connectivity
CC = sqrtm( W );

% Compute tuning curves after neuron loss (same as above)
for i=1:Nx
  
  x=[X10(i) nu0*BTV]';
  d  = CC \ (F*x);
  rp = lsqnonneg(CC,d);             
  
  % estimate representation error
  xestp=G*rp;
  Eafter(i)=mean((x - xestp).*(x - xestp))+QBeta*(rp'*rp);

end


%% --------------------------- PLOT RESULTS -----------------------------

% plotting parameters
LW=3.5;
LW2=2;
font='Helvetica';
Fontangle='normal';
Fontsize1=25;
Fontsize2=25;

%% --------------------------- FIGURE 6B --------------------------------
figure(61);clf;
set(gcf, 'Color', 'w');

% plot tuning curves
for i=1:Nf*2
    plot(X2(i,:),Rp2(i,:),'-k','MarkerSize', 15,'LineWidth', LW);
    hold on;
end
ylabel('Firing Rate (spikes/sec)','FontSize',Fontsize1,'FontAngle',Fontangle,'FontName',font);
xlabel('Ipsi. Eye Position','FontSize',Fontsize1,'FontAngle',Fontangle,'FontName',font);
box on;
axis([-1 1.05 0 102]);
set(gca,'YTick',0:10:100)
set(gca,'YTickLabel',0:10:100,'FontSize',Fontsize1,'FontAngle',Fontangle,'FontName',font)
set(gca,'FontSize',Fontsize2,'FontAngle',Fontangle,'FontName',font,'LineWidth', LW2)
set(gca,'TickDir','out')
axis square;

% plot inset - recruitment order
axes('pos',[0.32 0.65 0.2 0.2 ]);
hold on;
plot(Eth,kN,'.k','MarkerSize', 15,'LineWidth', LW);
ylabel('k_n','FontSize',Fontsize1,'FontAngle',Fontangle,'FontName',font);
xlabel('E_{th,n}','FontSize',Fontsize1,'FontAngle',Fontangle,'FontName',font);
box off;
axis([min(min(Eth),-2.5) 0.5 0 150]);
set(gca,'XTick',-2:1:0)
set(gca,'XTickLabel',{'-2','-1','0'},'FontSize',20,'FontAngle','Italic','FontName','TimesNewRoman')
set(gca,'YTick',0:50:150)
set(gca,'YTickLabel',{'0','50','100','150'},'FontSize',Fontsize1,'FontAngle',Fontangle,'FontName',font)
set(gca,'FontSize',Fontsize2,'FontAngle',Fontangle,'FontName',font,'LineWidth', LW2)
set(gca,'TickDir','out')

%% --------------------------- FIGURE 6D --------------------------------

figure(62); clf;
set( gcf, 'Color', 'w' );
plot(X10,(Eafter-Ebefore),'.k','MarkerSize',32);
hold on; plot([0 0],[-2 2],'-k','LineWidth',1);
hold on; plot([-2 2],[0 0],'-k','LineWidth',1);
xlabel('Eye Position (deg)','FontSize',Fontsize1,'FontAngle',Fontangle,'FontName',font);
ylabel('Delta E','FontSize',Fontsize1,'FontAngle',Fontangle,'FontName',font);
axis square;
axis([-1.25 1.25 -1.25 1.25])
set(gca,'YTick',-1:0.5:1,'YTickLabel',{'-1','-0.5','0','0.5','1',''},'LineWidth',1, 'TickDir', 'out','TickLength', [0.035,0.035],'FontSize',15 );
set(gca,'XTick',[-1,0,1],'XTickLabel',{'-1','0','1'},'LineWidth',1, 'TickDir', 'out','TickLength', [0.035,0.035],'FontSize',18 );
