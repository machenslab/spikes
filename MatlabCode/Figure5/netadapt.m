function [O,rO,V,x,xest,I,Wref,Gamma,thres,Ucc,mI]=netadapt(Il,Wref,Gamma,Nneurontot,Ninput,thres,Ucc,mI,nspeed,nlat)


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

I=Il;

x=zeros(Ninput,Ntime);
x(1,:)=0;
V=zeros(Nneurontot,Ntime);
O=zeros(Nneurontot,Ntime);
rO=zeros(Nneurontot,Ntime);
V(:,1)=randn(Nneurontot,1)*0.01;
rO(:,1)=0;
xest=zeros(Ninput,Ntime);
xest(1,:)=0; 
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
        if t>1000               
            Gamma=Gamma+nspeed*eps*((O(:,t+1))*((In(:,t+1)-100*Ucc'*O(:,t+1))'));
            
        end
    end

    rO(:,t+1)=(1-lambda*dt)*rO(:,t)+O(:,t+1);
    xest(:,t+1)=(1-lambda*dt)*xest(:,t)+Gamma'*O(:,t+1);                        
end


% Homeostatic regulation of threshold (firing rate maintained between 0.5 and 10Hz)

thres=thres+ nspeed*eps*(sum(O,2)>20);
thres=thres- nspeed*eps*(sum(O,2)<1);

   



