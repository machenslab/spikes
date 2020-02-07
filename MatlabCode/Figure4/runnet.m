function [rOE, OE, VE,rOI, OI, VI] = runnet(dt,Ntime, lambda,NneuronE,NneuronI,TE,TI,Input,FE,CEE,CEI,CII,CIE)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%
%%%% This function runs the network without learning. It take as an
%%%% argument the time step dt, the leak of the membrane potential lambda,
%%%% the Input of the network, the recurrent connectivity matrices and the feedforward
%%%% connectivity matrix F, the number of neurons Nneuron, the length of
%%%% the Input Ntime, and the Threhsold. It returns the spike trains O
%%%% the filterd spike trains rO, and the membrane potentials V.
%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



rOE=zeros(NneuronE,Ntime);
OE=zeros(NneuronE,Ntime);
VE=zeros(NneuronE,Ntime);
rOI=zeros(NneuronI,Ntime);
OI=zeros(NneuronI,Ntime);
VI=zeros(NneuronI,Ntime);

LSE=-ones(1,NneuronE);
LSI=-ones(1,NneuronI);

kE=1;
kI=1;

for t=2:Ntime
   
    VE(:,t)=(1-lambda*dt)*VE(:,t-1) + dt*FE'*Input(:,t-1)+OE(kE,t-1)*CEE(:,kE)+ OI(kI,t-1)*CIE(:,kI)+0.001*randn(NneuronE,1);
    
    [mE,kE]= max(VE(:,t) - TE-0.02*randn(NneuronE,1)); %choosing the neuron that has the biggest membrane potential int he E population
    
    rOE(:,t)=(1-lambda*dt)*rOE(:,t-1); % the filterd spike train has a leak of lambda
    
    if (mE>=0 && LSE(1,kE)<0 ) %if the voltage of this neuron is greater than the threshold and his refratory period is over it spikes
        
        LSE(1,kE)=10;     % the refractory period of this neuron is set to 10 time steps   
        OE(kE,t)=1;       % the spike vatiable of this neuron at time t is equal to one
        rOE(kE,t)=rOE(kE,t)+1; % the filtered spike train of the is incremented to one
        
    end
    
    LSE=LSE-1; %the refreactory period of all neurons is decremented
    
    
    VI(:,t)=(1-lambda*dt)*VI(:,t-1)+OE(kE,t)*CEI(:,kE)+OI(kI,t-1)*CII(:,kI)+0.002*randn(NneuronI,1);
    
    [mI,kI]= max(VI(:,t) - TI-0.02*randn(NneuronI,1));  %choosing the neuron that has the biggest membrane potential int he I population
    
    rOI(:,t)=(1-lambda*dt)*rOI(:,t-1) ;
    
    if (mI>=0 && LSI(1,kI)<0)
        
        LSI(1,kI)=10;
        OI(kI,t)=1;
        rOI(kI,t)=rOI(kI,t)+1;
        
    end
    
    LSI=LSI-1;
    
    
end




