% This program produces Figure 8 and 9 for "Optimal Compensation for
% Neuron Loss" by David Barrett, Sophie Deneve, and Christian Machens
%
% Uses Optimization toolbox from Mathworks

clear all;
rng( 1211 );
options = optimset('Display', 'off','MaxIter',500,'LargeScale','on', ...
                   'TolFun',0.000000001,'TolPCG',0.1);

%============================== FIGURE 8 ===================================

%--------------------------- FREE PARAMETERS -------------------------------

Ntarget = 288;                      % Number of neurons in network (N>=288)
lam=0.005;                          % Decoder timescale (inverse msec)

% By default, we work with the N = 288 neurons with decoders taken
% from the file 'Weights.mat' (see below). The network size can be 
% increased artifically by randomly shifting the decoding weights
% This will happen if Ntarget > 288

%--------------------------- LOAD DATA -------------------------------------

load('Weights');                    % variables W and nextsweep
Nneurons = size(W,2);               % Default network size
Nz = size(W,1);                     % dimensionality of image patches
Beta=Nz/Nneurons/2;                 % Constant firing rate cost

%---------------------------- REDUNCANCY GENERATOR -------------------------

if Ntarget > Nneurons,
  Ndiff = Ntarget - Nneurons;
  D = zeros( Nz, Ntarget );
  D( :, 1:Nneurons ) = W;
  for k=1:Ndiff
    no = randi(Nneurons,1);
    fld = reshape( W(:,no), 12, 12 );
    xind = circshift( (1:12)', randi(12,1) );
    fld = fld(xind, :);
    yind = circshift( (1:12)', randi(12,1) );
    fld = fld(:,yind, :);
    D(:,Nneurons+k) = fld(:);
  end
  Nneurons = Ntarget;               % new number of neurons
  W = D;
end

%--------------------------- STIMULI / SIGNALS ------------------------------

% Gabor Filters - spatial offsets
Nx=20;
X_max=1.25;
X_min=-1.25;
X_all=(0:Nx-1)/(Nx-1)*(X_max-X_min)+X_min;

% Gabor Filters - angles
Ntheta=17;
theta_max=pi;
theta_min=-pi;
theta_all=(0:Ntheta-1)/(Ntheta-1)*(theta_max-theta_min)+theta_min;
% careful, last theta_all identical to first one !

%--------------------------- TUNING CURVES IN INTACT STATE ------------------

% connectivity for QP algorithm
H=2*W'*W;
d1=Beta*(ones(1,Nneurons));

% compute preferred tuning for all neurons for Nthta different
% centers using gabor functions
disp('Calculating firing rates in intact system ...');
frate = zeros( Ntheta, Nx, Nneurons );
for k=1:Ntheta
  for i=1:Nx
    x = X_all(i) * cos( theta_all(k) );
    y = X_all(i) * sin( theta_all(k) );
    gb = 10 * gabor_fn(0.2,theta_all(k),1,pi/2,0.2,x,y);
    xs=reshape( gb, Nz, 1 );
    d=d1-2*xs'*W;
    [f,E] = quadprog(H,d,[],[],[],[],zeros(Nneurons,1),...
                     ones(Nneurons,1)*inf,zeros(Nneurons,1),options);
    frate(k,i,:)=f;
    
    % compute incurred errors
    xsest = W*squeeze(frate(k,i,:));
    error(k,i) = norm( xs-xsest );
  end
  F(k,:)     = max(frate(k,:,:));
  Fmean(k,:) = mean(frate(k,:,:));
end

% take maximum at preferred direction (this should be done quicker!)
for i=1:Nneurons
  [sortedori,labelsori]=sort((F(:,i)),'descend');
  prefdir(i)=theta_all(labelsori(1));
end

%--------------------------- KNOCK OUT NEURONS ------------------------------

% compute which neurons to knock out
Knockout    =   0;
Knockoutmax =  Knockout + pi/12;
Knockoutmin =  Knockout - pi/12;
k=0;
for i=1:Nneurons   
  if ( (prefdir(i) < Knockoutmax & prefdir(i) > Knockoutmin) ),
    k=k+1;
    knockouts(k)=i;
  end
end
Totalknockout = k;
index = randperm(round(Totalknockout*0.5));
knockouts = knockouts(index);
Totalknockout = length(knockouts);

%--------------------------- TUNING CURVES OF K.O. NEURONS -----------------

% multi-unit activity measured from k.o. neurons before k.o.
FMUko = mean( F(:,knockouts),2 );

%--------------------------- TUNING CURVES IN K.O. STATE -------------------

% compute new decoder (W) and copy uncompensated firing rates
Wknockout = W;
Funcompensated = F;
for i=1:Totalknockout
    Wknockout(:,knockouts(i)) = zeros(Nz,1);
    Funcompensated(:,knockouts(i)) = zeros(Ntheta,1);
end

% compute new connectivity
H=2*Wknockout'*Wknockout;
d1=Beta*(ones(1,Nneurons));

% recompute firing rates and measure direct tuning
disp('Calculating firing rates in k.o. system ...');
frateko = zeros( Ntheta, Nx, Nneurons );
for k=1:Ntheta
  for i=1:Nx
    x=X_all(i)*cos(theta_all(k));
    y=X_all(i)*sin(theta_all(k));
    gb=10*gabor_fn(0.2,theta_all(k),1,pi/2,0.2,x,y);
    xs=reshape( gb, Nz, 1 );
    d=d1-2*xs'*Wknockout;
    [f,E] = quadprog(H,d,[],[],[],[],zeros(Nneurons,1),...
                     ones(Nneurons,1)*inf,zeros(Nneurons,1),options);
    frateko(k,i,:)=f;
    
    % compute errors for compensated or uncompensated cases
    xsestko = Wknockout*squeeze(frateko(k,i,:));
    xsestuncomp = Wknockout*squeeze(frate(k,i,:));
    errorko(k,i) = norm( xs-xsestko );
    erroruncomp(k,i) = norm( xs-xsestuncomp );
  end
  Fko(k,:)=max(frateko(k,:,:));
  Fkomean(k,:)=mean(frateko(k,:,:));
end

% take maximum at preferred direct (this should be done quicker!)
for i=1:Nneurons
  [sortedori,labelsori]=sort((Fko(:,i)),'descend');
  prefdirko(i)=theta_all(labelsori(1));    
end

%--------------------------- DIFFERENT INTACT / K.O. -------------------

% compute difference in direct tuning
dif=prefdirko- prefdir;
for i=1:Nneurons
    if dif(i) > pi
       dif(i)=dif(i)-2*pi ;
    end
    if dif(i) < -pi
       dif(i)=dif(i)+2*pi ;
    end
end

% Resort neurons to cluster k.o. neurons
intact   = setxor(1:Nneurons, knockouts);
[s, ind] = sort( prefdir(intact) );
nbet     = round(length(ind)/2);
all = [intact(ind(1:nbet)), knockouts, intact(ind(nbet+1:end)) ];

% Compute Firing rate differences
dF  = Fko(:,all) - F(:,all);
dF(:,(nbet+1):(nbet+Totalknockout))=NaN; % black out k.o. neurons

%--------------------------- PLOT FIGURE 8 ---------------------------------

figure(8); clf;
set(gcf, 'Color', 'w');

% readjust units to scale for firing rates in Hz
Fmean   = Fmean * (lam*1000);
Fkomean = Fkomean * (lam*1000);
F       = F * (lam*1000);
Fko     = Fko*(lam*1000);
Funcompensated = Funcompensated * (lam*1000);
FMUko   = FMUko * (lam*1000);

% Plotting parameters
angles = 180*theta_all/pi;          % radiant -> degree
w = 0.35;                           % width of plots

% Panel A
ax =axes('Position', [0.09 0.51 w 0.4] );
imagesc( angles, 1:Nneurons, F(:,all)' );
caxis([0 100]);
colormap('hot');
set( gca, 'TickDir', 'out' );
set( gca, 'XTick', [-180 -90 0 90 180]);
set( gca, 'YDir', 'normal' );
xlabel('Stimulus Direction');
ylabel('Neuron #');
cbax = colorbar('peer',ax,'North');
set( cbax, 'Position', [0.29 0.9 0.15, 0.01] );
set( cbax, 'XAxisLocation', 'top' );
set( ax, 'Position', [0.09 0.51 w 0.38] );

% Panel B
ax = axes('Position', [0.58 0.51 w 0.4] );
imagesc( angles,1:Nneurons, dF' );
caxis([-20 100]);
set( gca, 'TickDir', 'out' );
set( gca, 'XTick', [-180 -90 0 90 180]);
set( gca, 'YTick',  Nneurons-250:50:Nneurons, 'YTickLabel', {'', '', '', '', '', ''} );
xlabel('Stimulus Direction');
ylabel('Neuron #');
colormap('hot');
cbax = colorbar('peer',ax,'North');
set( cbax, 'Position', [0.78 0.9 0.15, 0.01] );
set( cbax, 'XAxisLocation', 'top' );
set( ax, 'Position', [0.58 0.51 w 0.38] );

% Panel D
axes('Position', [0.58 0.25 w 0.175] );
hold on;
plot( angles, mean(Fko'-Funcompensated'),'.-k','MarkerSize', 15,'LineWidth', 1.5);
ylabel('DF (Hz)');
xlabel('Stimulus Direction');
box off;
axis([-185, 180, -0.1, 40]);
set(gca,'XTick',[-180 -90 0 90 180], 'YTick',[0 20 40] );
set(gca,'LineWidth',1, 'TickDir', 'out'  );
set(gca,'ticklength',2*get(gca,'ticklength'))

% Panel D; upper left histogram
axes('Position', [0.53 0.12 0.15 0.055] );
hist( dF(1,:), -20:10:80 );
axis([-20 80 0 50] );
set( gca, 'XTick', [0 50], 'TickDir', 'out', 'YTick', [0 40] );
set( gca, 'XTickLabel', {'',''}, 'YTickLabel', {'',''});
set(gca,'ticklength',3*get(gca,'ticklength'))
box off;

% Panel D; upper right histogram
axes('Position', [0.73 0.12 0.15 0.055] );
hist( dF(9,:), -20:10:80 );
axis([-20 80 0 50] );
set( gca, 'XTick', [0 50], 'TickDir', 'out', 'YTick', [0 40] );
set( gca, 'XTickLabel', {'',''}, 'YTickLabel', {'',''});
set(gca,'ticklength',3*get(gca,'ticklength'))
box off;

% Panel D; lower left histogram
axes('Position', [0.65 0.04 0.15 0.055] );
hist( dF(5,:), -20:10:80 );
axis([-20 80 0 50] );
set( gca, 'XTick', [0 50], 'TickDir', 'out', 'YTick', [0 40] );
xlabel('DF (Hz)');
ylabel('# neurons');
set(gca,'ticklength',3*get(gca,'ticklength'))
box off;

% Panel D; lower right histogram
axes('Position', [0.85 0.04 0.15 0.055] );
hist( dF(13,:), -20:10:80 );
axis([-20 80 0 50] );
set( gca, 'XTick', [0 50], 'TickDir', 'out', 'YTick', [0 40] );
set( gca, 'XTickLabel', {'',''}, 'YTickLabel', {'',''});
set(gca,'ticklength',3*get(gca,'ticklength'))
box off;

axes('Position', [0.09 0.25 w 0.175] );
hold on;
plot(180*prefdir/pi, 180*dif/pi,'.k','MarkerSize', 15);
xlabel('Preferred Direction (PD)');
ylabel('DTh');
box off;
axis([-185, 180, -181, 181]);
set(gca,'XTick',[-180 -90 0 90 180], 'YTick',[-180 0 180] );
set(gca,'LineWidth',1, 'TickDir', 'out'  );
set(gca,'ticklength',2*get(gca,'ticklength'))

axes('Position', [0.09 0.06 0.3 0.1] );
hold on;
errratio = erroruncomp ./ errorko;
hist( errratio(:), 0:0.2:3 );
axis([0 3 0 250] );
xlabel('Error ratio');
ylabel('# stimuli' );
set( gca, 'TickDir', 'out');
set(gca,'ticklength',2*get(gca,'ticklength'))

%============================== FIGURE 9B ===================================

% (Will only reproduce the plot if Ntarget=288 at beginning)

figure(9);clf;
set(gcf, 'Color', 'w');

% Plot three example neurons; tuning curves are rotated
% to match preferred tuning of k.o. neurons
rot(1) = pi/6;
rot(2) = 5*pi/6;
rot(3) = 5*pi/6;

k=1;
for neuron=[155 140 284];         % selected example neurons

  % Plot bulk tuning of k.o. neurons
  subplot( 3,3,3*k-2 );
  orplot( theta_all-rot(k), FMUko, 65 );
  
  % Plot tuning before k.o.
  subplot( 3,3,3*k-1 );
  size1=ceil(max(max(F(:,neuron))));
  size2=ceil(max(max(Fko(:,neuron))));
  size1 = max(size1,size2);
  size1=size1+(5-mod(size1,5));
  orplot( theta_all-rot(k), F(:,neuron), size1 );
  
  % Plot tuning after k.o.
  subplot( 3,3,3*k );
  orplot( theta_all-rot(k), Fko(:,neuron), size1 );
  
  k = k+1;
end
