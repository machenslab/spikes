function [O,rO,V,x,xest,I,Wref,Gamma,thres,Ucc,mI]=netall(SPu,Wref,Gamma,Nneurontot,Ninput,thres,Ucc,mI,nspeed,nlat)

% nlat and nspeed parameterize the learning rates of feedforward and lateral connections

% Number of dt in input signal
Ntime=49901;

%time step = 1/20 ms
dt=0.00005;

%decoder leak rate = 8Hz
lambda=8;

%leak for estimating the covariance of input/output currents
la=0.2;

%baseline learning rate
eps=0.01;

%cost terms
mu=0.1;
alpha=1;

%Leak for FF learning = 1000 Hz
ep=2000;

%gain of input current
gain=0.03;


%Generate input current. Note that to get x=SP, we need a current input dot(SP) + lambda SP

Un=randperm(230500);
cI=zeros(Ninput,49901);
I=zeros(Ninput,49901);

% Interpolate and differentiate speech to obtain input signal 

for i=1:Ninput
    SPu(i,Un(1)+1:Un(1)+5)=0;    
    cI(i,:)=(SPu(i,Un(1)+1)*100+(cumsum(interp1([1:500],gradient(SPu(i,Un(1)+1:Un(1)+500)),[1:0.01:500]))))*0.1;
    % input is such that when filtered by e^-(lambda t), you get back the speech signal 
    I(i,:)=gradient(cI(i,:))+lambda*dt*cI(i,:);
end

I=I*gain;

% filtered input signal ( = interpolated speech)
x=zeros(Ninput,Ntime);
x(1,:)=0;

%Membrane potentials
V=zeros(Nneurontot,Ntime);

%spike trains
O=zeros(Nneurontot,Ntime);

%filtered spike trains
rO=zeros(Nneurontot,Ntime);

%output estimate
xest=zeros(Ninput,Ntime);

%Input current interg
In=zeros(Ninput,Ntime);
BN=randn(Nneurontot,Ntime)*0.005;

for t=1:49900
        
    W=-Wref;
    x(:,t+1)=(1-lambda*dt)*x(:,t)+I(:,t+1);
    In(:,t+1)=(1-ep*dt)*In(:,t)+ep*dt*I(:,t+1);
    
    
         
    Input=Gamma*I(:,t+1);       
    V(:,t+1)=(1-lambda*dt)*V(:,t)+Input+W*O(:,t);          
    crit=(V(:,t+1)+BN(:,t+1))-thres;         
    O(:,t+1)=(crit>0);
    mI=(1-la*dt)*mI+la*dt*In(:,t); 

    Ucc=(1-la*dt)*Ucc+la*dt*((Gamma*(In(:,t+1)-mI))*(In(:,t+1)-mI)');
    

 
        
    if (sum(O(:,t+1))>1)           
        [~,v]=max(crit);          
        O(:,t+1)=0;           
        O(v,t+1)=1;        
    end
    
    if (sum(O(:,t+1))>0)                  
        if t>1
            Wref=Wref+nlat*eps*((O(:,t+1)*(alpha*(V(:,t+1)+mu*rO(:,t)))')'+(ones(Nneurontot,1)*O(:,t+1)').*(-Wref+mu*diag(ones(1,Nneurontot))));            
        end        
        if t>5000
            % To learn the feedforward connections, we used alpha = 1/120. 
            Gamma=Gamma+nspeed*eps*120*((O(:,t+1))*((1/120*In(:,t+1)-Ucc'*O(:,t+1))'));
            
        end
    end

    rO(:,t+1)=(1-lambda*dt)*rO(:,t)+O(:,t+1);
    xest(:,t+1)=(1-lambda*dt)*xest(:,t)+Gamma'*O(:,t+1);                        
end


% Homeostatic regulation of threshold (maintain mean firing rate between 0.5 and 15Hz)

thres=thres+ nspeed*eps*(sum(O,2)>20);
thres=thres- nspeed*eps*(sum(O,2)<1);

   



