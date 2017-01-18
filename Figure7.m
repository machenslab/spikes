% This program produces Figure 7 for "Optimal Compensation for
% Neuron Loss" by David Barrett, Sophie Deneve, and Christian Machens
%
% (This takes more than an hour on a standard laptop in 2016)
% Uses Optimization toolbox from Mathworks

rng('default');
options = optimset('Display', 'off','MaxIter',500,'LargeScale','on', ...
     'TolFun',0.000000001,'TolPCG',0.1);

%=============================== FIGURE 7 ================================

%------------------------------- LOAD DATA -------------------------------

% Load original image
w = 1536; h = 1024;              % size of the image
f1 = fopen( 'img18.imc', 'rb', 'ieee-be' );
I  = fread( f1, [w,h], 'uint16' );
fclose( f1 );

% Only use part of the image
w = 164; h = 128;                % Width & Height
a = 15;  b = 1;                  % Upper left corner
I1=I(a+1:w+a,b+1:h+b)/10;

% Load decoder weights
load('Weights');
Nz   = size(W,1);                % Size of Signal Space
Nneurons = size(W,2);            % network size
Beta = Nz/Nneurons/2;            % Firing rate cost

%--------------------------- STIMULI / SIGNALS ------------------------------

% Gabor Filters - spatial offsets
Nx    = 20; 
X_max =  1.25;
X_min = -1.25;
X_all = (0:Nx-1)/(Nx-1)*(X_max-X_min)+X_min;

% Gabor Filters - angles
Ntheta    = 17;
theta_max = pi;
theta_min = -pi;
theta_all = (0:Ntheta-1)/(Ntheta-1)*(theta_max-theta_min)+theta_min;
% careful, last theta_all identical to first one !

% Image patches, sampled from original image with overlap
overlap = 2;
k = 0;
for i=1:(w-12)/overlap+1
  for j=1:(h-12)/overlap+1
    k = k+1;
    x = ((i-1)*overlap+1):(i-1)*overlap+12;
    y = ((j-1)*overlap+1):(j-1)*overlap+12;
    I2(k,:) = reshape(I1(x,y),144,1);
  end
end
Nsamples=k;

%--------------------------- PREFERRED TUNING ------------------------------

% newtork connectivity and costs, prepared for QP algorithm
H  = 2*W'*W;
d1 = Beta*(ones(1,Nneurons));

% compute preferred tuning for all neurons for Ntheta different
% angles using gabor functions
disp('Calculating firing rates in intact system ...');
frate = zeros( Ntheta, Nx, Nneurons );
for k=1:Ntheta
  for i=1:Nx
    x  = X_all(i) * cos( theta_all(k) );
    y  = X_all(i) * sin( theta_all(k) );
    gb = 10 * gabor_fn(0.2,theta_all(k),1,pi/2,0.2,x,y);%figure; imagesc(gb);
    xs=reshape( gb, Nz, 1 );
    d=d1-2*xs'*W;
    [f,E] = quadprog(H,d,[],[],[],[],zeros(Nneurons,1),...
                     ones(Nneurons,1)*inf,zeros(Nneurons,1),options);
    frate(k,i,:)=f;
  end
  F(k,:) = max(frate(k,:,:));
end

% take maximum at preferred direction (this should be done quicker!)
for i=1:Nneurons
    [sortedori,labelsori]=sort((F(:,i)),'descend');
    pref_directF(i)=theta_all(labelsori(1));
end

%--------------------------- KNOCK OUT NEURONS ------------------------------

% compute which neurons to knock out
k=0;
for i=1:Nneurons   
  knocked=0;
  if pref_directF(i) ==pi/2 && knocked==0 && i~=104 && i~=248
    k=k+1;
    knockouts(k)=i;
    knocked=1;    
  end
  if pref_directF(i) ==pi/2+pi/8 && knocked==0 && i~=104 && i~=248
    k=k+1;
    knockouts(k)=i;
    knocked=1;    
  end  
  if pref_directF(i) ==pi/2-pi/8 && knocked==0 && i~=104 && i~=248
    k=k+1;
    knockouts(k)=i;
    knocked=1;    
  end
  if pref_directF(i) ==-pi/2 && knocked==0 && i~=104 && i~=248
    k=k+1;
    knockouts(k)=i;
    knocked=1;    
  end
  if pref_directF(i) ==-pi/2+pi/8 && knocked==0 && i~=104 && i~=248
    k=k+1;
    knockouts(k)=i;
    knocked=1;    
  end
  if pref_directF(i) ==-pi/2-pi/8 && knocked==0 && i~=104 && i~=248
    k=k+1;
    knockouts(k)=i;
    knocked=1;    
  end
end
Totalknockout=k;

% compute decoder for knocked out case 
Wknockout=W;
for i=1:Totalknockout
  Wknockout(:,knockouts(i))=zeros(1,Nneurons/2);
end

%--------------------------- RESPONSE TO IMAGE PATCHES -----------------------

% Network connectivity for original and k.o. case
H=2*W'*W;                        % Original Connectivity
Hko=2*Wknockout'*Wknockout;      % K.o Connectivity
d1=Beta*(ones(1,Nneurons));      % Cost term

% Compute firing rates for all image patches
fprintf('Number of steps= %ld \n',Nsamples);
for i=1:Nsamples
  i
  % Input signal
  xs=I2(i,:)';
  
  % QP of firing rates for intact case
  d=d1-2*xs'*W;
  [f,E] = quadprog(H,d,[],[],[],[],zeros(Nneurons,1),...
                   ones(Nneurons,1)*inf,zeros(Nneurons,1),options);

  % QP of firing rates for k.o. case
  d=d1-2*xs'*Wknockout;
  [fko,E] = quadprog(Hko,d,[],[],[],[],zeros(Nneurons,1),...
                     ones(Nneurons,1)*inf,zeros(Nneurons,1),options);

  % Compute estimates of image patches
  xm(i,:)      = W*f;            % Original
  xmko(i,:)    = Wknockout*f;    % Uncompensated
  xmoptko(i,:) = Wknockout*fko;  % Compensated
end

%--------------------------- IMAGE RECONSTRUCTION ---------------------------

Im     =  zeros(w,h);            % Original
Imko   =  zeros(w,h);            % Uncompensated
Imoptko = zeros(w,h);            % Compensated
m=0;
for i=1:(w-12)/overlap+1
  for j=1:(h-12)/overlap+1
    m=m+1;
    x=((i-1)*overlap+1):(i-1)*overlap+12;
    y=((j-1)*overlap+1):(j-1)*overlap+12;

    % Original Image Reconstruction
    Imhold      = reshape(xm(m,:),12,12); 
    Im(x,y)     = Im(x,y) + Imhold;

    % Uncompensated Image Reconstruction
    Imkohold    = reshape(xmko(m,:),12,12); 
    Imko(x,y)   = Imko(x,y) + Imkohold;

    % Compensated Image Reconstruction
    Imoptkohold = reshape(xmoptko(m,:),12,12); 
    Imoptko(x,y)= Imoptko(x,y) + Imoptkohold;    
  end
end

%--------------------------- PLOT RESULTS ----------------------------------

% Plot original Image
figure(71);
set(gcf, 'Color', 'w');
colormap( gray );
imagesc( I' );
set( gca, 'Visible', 'off' );

% Plot Image and its Reconstructions
figure(72); clf;
set(gcf, 'Color', 'w');

% Original Image
axes('pos',[0.05 0.68 0.4 0.4*h/w ]);
colormap( gray );
imagesc( I1' );
set( gca, 'Visible', 'off' );

% Image, as reconstructed from full population, not edge corrected
axes('pos',[0.55 0.68 0.4 0.4*(h-24)/(w-24) ]);
colormap( gray );
imagesc( Im' );
set( gca, 'Visible', 'off' );

% Image, reconstructed after k.o., without compensation
axes('pos',[0.55 0.36 0.4 0.4*(h-24)/(w-24) ]);
colormap( gray );
imagesc( Imko' );
set( gca, 'Visible', 'off' );

% Image, reconstructed after k.o., with compensation
axes('pos',[0.55 0.04 0.4 0.4*(h-24)/(w-24) ]);
colormap( gray );
imagesc( Imoptko' );
set( gca, 'Visible', 'off' );

% Plot subsample of decoding vectors
figure(73); clf;
set(gcf, 'Color', 'w');
startpos=25;
i=0;
for k=1:6
  for m=1:3
    
    i=i+1;
    x=(m-1)/3+0.01;
    y=(k-1)/12+0.01;
    
    axes('pos',[x y+0.5 1/3 1/14 ],'FontSize',8);
    imagesc( reshape( W(:,i+startpos), 12, 12) );
    colormap( 'gray' );
    set( gca, 'Visible', 'off' );
    axis square;
    
    axes('pos',[x y 1/3 1/14 ],'FontSize',8);
    imagesc( reshape( W(:,i+144+startpos), 12, 12)' ); 
    colormap( 'gray' );
    set( gca, 'Visible', 'off' );
    axis square;
    
  end
end

% Plot k.o. decoding vectors
figure(74); clf;
set(gcf, 'Color', 'w');
i=0;
for k=1:7
    for m=1:3
      
        i=i+1;
        x=(m-1)/3+0.01;
        y=(k-1)/7+0.01;
        
        axes('pos',[x y 1/3 1/8 ],'FontSize',8);
        imagesc( reshape( W(:,knockouts(i)), 12, 12)' );
        colormap( 'gray' );
        set( gca, 'Visible', 'off' );
        axis square;
    
    end
end





