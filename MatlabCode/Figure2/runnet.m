function [rO, O, V] = runnet(dt, lambda, F ,Input, C,Nneuron,Ntime, Thresh)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%
%%%% This function runs the network without learning. It take as an
%%%% argument the time step dt, the leak of the membrane potential lambda,
%%%% the Input of the network, the recurrent connectivity matrix C, the feedforward
%%%% connectivity matrix F, the number of neurons Nneuron, the length of
%%%% the Input Ntime, and the Threhsold. It returns the spike trains O
%%%% the filterd spike trains rO, and the membrane potentials V.
%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rO=zeros(Nneuron,Ntime);%filtered spike trains
O=zeros(Nneuron,Ntime); %spike trains array
V=zeros(Nneuron,Ntime); %mamebrane poterial array

for t=2:Ntime

    V(:,t)=(1-lambda*dt)*V(:,t-1)+dt*F'*Input(:,t-1)+C*O(:,t-1)+0.001*randn(Nneuron,1);%the membrane potential is a leaky integration of the feedforward input and the spikes

 
    [m,k]= max(V(:,t) - Thresh-0.01*randn(Nneuron,1));%finding the neuron with largest membrane potential
        
    if (m>=0)  %if its membrane potential exceeds the threshold the neuron k spikes  
        O(k,t)=1; % the spike ariable is turned to one
    end

    rO(:,t)=(1-lambda*dt)*rO(:,t-1)+1*O(:,t); %filtering the spikes
    
end

end




