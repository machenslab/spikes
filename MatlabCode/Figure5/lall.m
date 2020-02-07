% Number of neurons
Nneurontot=100;

% Number of frequency channels
Ninput=25;

% Number of learning iterations
Nit=8000;

% load speech stimuli
load speech;
SPu=speech.data;

% Set it to 0 if you want to result learning, 1 if you want to learn from
% scratch. 

reset=1;

if reset==1
    % Initialize feedforward weights "F"
    Gamma(:,1:Ninput)=randn(Nneurontot,Ninput)*0.025;
    
    % Initialize recurrent connections
    Wref=(randn(Nneurontot,Nneurontot)*0.1.*(1-diag(ones(1,Nneurontot))))+diag(ones(1,Nneurontot))*0.8;
    
    % Initiamize thresholds
    thres=0.5;
    
    % Initialize correlation matrix of pre and postsynaptic currents
    Ucc=0.0001;
    
    % Initialize mean input currents
    mI=zeros(Ninput,1);

end


k=1;

%Run learning algorithm with decreasing learning rates. 

for it=1:Nit
    it
    if it<5
        [O,rO,V,x,xe,I,Wref,Gamma,thres,Ucc,mI]=netall(SPu,Wref,Gamma,Nneurontot,Ninput,thres,Ucc,mI,0,0);
    elseif it>4 && it<2000
        [O,rO,V,x,xe,I,Wref,Gamma,thres,Ucc,mI]=netall(SPu,Wref,Gamma,Nneurontot,Ninput,thres,Ucc,mI,5,5);
    elseif it>1999 && it<4000
         [O,rO,V,x,xe,I,Wref,Gamma,thres,Ucc,mI]=netall(SPu,Wref,Gamma,Nneurontot,Ninput,thres,Ucc,mI,2,2);
     elseif it>3999 it<6000
         [O,rO,V,x,xe,I,Wref,Gamma,thres,Ucc,mI]=netall(SPu,Wref,Gamma,Nneurontot,Ninput,thres,Ucc,mI,0.5,0.5);
     elseif it>5999
         [O,rO,V,x,xe,I,Wref,Gamma,thres,Ucc,mI]=netall(SPu,Wref,Gamma,Nneurontot,Ninput,thres,Ucc,mI,0.1,0.1);
    end 
    
    %Save the learnt weights and thresholds every 500 time steps. 

    if it>k*500
        save Icurrent Gamma Wref thres mI Ucc
        k=k+1;
    end

    %estimate network decoder
    D=pinv(Gamma,0.1)*(Wref);
    
    %compute error
    xest=D*rO;
    error=mean(std(xest(:,30000:49901)'-x(:,30000:49901)'))/mean(std(x'));
    er=mean(std(xest(:,30000:49901)'-x(:,30000:49901)'));

    
    %Show results
    [sum(sum(O)),sum(sum(I.^2).^0.5)/(er*2*2^0.5),sum(sum(O')>0),error*100,max(max(Gamma))*100]

end

