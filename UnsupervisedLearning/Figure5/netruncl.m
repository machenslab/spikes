function [O,rO,V,x]=netruncl(I,Wref,Gamma,Nneurontot,Ninput,thres,irec)


Ntime=length(I);


%time step = 1/20 ms
dt=0.00005;

%decoder leak rate = 8Hz
lambda=8;


x=zeros(Ninput,Ntime);
x(1,:)=0;
V=zeros(Nneurontot,Ntime);
O=zeros(Nneurontot,Ntime);
rO=zeros(Nneurontot,Ntime);
V(:,1)=randn(Nneurontot,1)*0.01;
rO(:,1)=0;
xest=zeros(Ninput,Ntime);
xest(1,:)=0; 
BN=randn(Nneurontot,Ntime)*0.005;

thres(irec)=1000;

for t=1:Ntime-1
        
    W=-Wref;
    x(:,t+1)=(1-lambda*dt)*x(:,t)+I(:,t+1);         
    Input=Gamma*I(:,t+1);       
    V(:,t+1)=(1-lambda*dt)*V(:,t)+Input+W*O(:,t);
    crit=(V(:,t+1)+BN(:,t+1))-thres;         
    O(:,t+1)=(crit>0);

        
    if (sum(O(:,t+1))>1)           
        [~,v]=max(crit);          
        O(:,t+1)=0;           
        O(v,t+1)=1;        
    end
    

    rO(:,t+1)=(1-lambda*dt)*rO(:,t)+O(:,t+1);
    xest(:,t+1)=(1-lambda*dt)*xest(:,t)+Gamma'*O(:,t+1);                        
end

   



