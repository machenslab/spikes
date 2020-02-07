function [FsE,CsEE,CsEI,CsII,CsIE,DecsE,DecsI,OEB,OIB,TimeT,T]=Learning(Nx,dt,lambda,epsr,epsf, mu, NneuronE,NneuronI, ThreshE,ThreshI,isiE,isiI, FE,CEE,CEI,CII,CIE)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%
%%%   This function  performs the learning of the
%%%%  recurrent and feedforward connectivity matrices.
%%%%
%%%%
%%%%  it takes as an argument the time step ,dt, the membrane leak, lambda,
%%%%  the learning rate of the feedforward and the recurrent
%%%%  conections epsf and epsr, the scaling parameters alpha and beta of
%%%%  the weights, mu the quadratic cost, the number of neurons on the
%%%%  Excitatory and inhibitory population the dimension of the input, the threshold of
%%%%  the neurons  an the initial feedforward and recuurrent connectivities
%%%%  FE,CEE,CEI,CII,CIE,
%%%%
%%%%   The output of this function are arrays, FsE,CsEE,CsEI,CsII,CsIE, containning the
%%%%   connectivity matrices sampled at exponential time instances. DecsE and DecI are the
%%%%   Optimal decoders for each instance of the registered c
%%%%   connectivities.
%%%%
%%%%   It also produces a figure. that represents the performace of the network
%%%%    through learning. And plots of the decoding anf the rasters before
%%%%    and after learning
%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%   Learning the optinal connectivities  %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


Nit=2^20 ;  %number of iterations
Ntime=1000; %number of steps in an iteration
TotTime=Nit*Ntime;%total time of Learning
T=floor(log(TotTime)/log(2)); %Computing the size of the matrix where the weights are stocked on times defined on an exponential scale

CsEE=zeros(T,NneuronE, NneuronE);
CsEI=zeros(T,NneuronI, NneuronE);
CsIE=zeros(T,NneuronE, NneuronI);
CsII=zeros(T,NneuronI, NneuronI);
FsE=zeros(T,Nx, NneuronE);


OE=0;                  % variable representing the presence of a spike in the E pop at a time step
VE=zeros(NneuronE,1);  % vector of membrane potentials of the E pop
kE=1;                  % index of the neuron that spiked in the E pop
rOE=zeros(NneuronE,1);

OI=0;                  % variable representing the presence of a spike in the I pop at a time step
VI=zeros(NneuronI,1);  % vector of membrane potentials of the I pop
kI=1;                  % index of the neuron that spiked in the I pop
rOI=zeros(NneuronI ,1);

ibarE=zeros(Nx,1);       %integration variable of the feedforward input to the E population used in the XE plastcity rule
ibarI=zeros(NneuronE,1); %integration constant of the spiking input to the I population fom the E population used in the EI plasticity rule

LSE=-ones(1,NneuronE);   %countdown variable used to set a refractory period for the  E neurons
LSI=-ones(1,NneuronI);   %countdown variable used to set a refractory period for the  I neurons

j=1;
l=1;

sigma=abs(30);                                                    %std of the gaussian smoothing window for the input
w=(1/(sigma*sqrt(2*pi)))* exp(-(([1:1000]-500).^2)/(2*sigma.^2)); %gaussian smoothing kernel for the input
w=w/sum(w);                                                       %normalization of the smoothing kernel
A=2000;                                                           %Amplitude of the input

Id=eye(NneuronI);

fprintf('%d percent of the learning  completed\n',0)

for i=2:TotTime
    
    if ((i/TotTime)>(j/100))
        fprintf('%d percent of the learning  completed\n',j)
        j=j+1;
    end
    
    if(mod(i,2^(l-1))==0)
        
        CsEE(l,:,:)=CEE;
        CsIE(l,:,:)=CIE;
        CsII(l,:,:)=CII;
        CsEI(l,:,:)=CEI;
        FsE(l,:,:)=FE;
        l=l+1;
    end
    
    
    if (mod(i-2,Ntime)==0) %Generating a new iput sequence every Ntime time steps
        Input=(mvnrnd(zeros(1,Nx),eye(Nx),Ntime))'; %generating a new sequence of input which a gaussion vector
        for d=1:Nx
            Input(d,:)=A*conv(Input(d,:),w,'same'); %smoothing the previously generated white noise with the gaussian window w
        end
    end
    
    VE=(1-lambda*dt)*VE + dt*FE'*Input(:,mod(i,Ntime)+1)+ OE*CEE(:,kE)+ OI*CIE(:,kI)+0.002*randn(NneuronE,1); %the Voltages of the E neurons integrate the Input and the spikes from the E and I neurons
    
    
    ibarE=(1-isiE*dt)*ibarE+dt*1*Input(:,mod(i,Ntime)+1); %integration of the Input in a short time window defined bu isiE
    ibarI=(1-isiI*dt)*ibarI;
    ibarI(kE,1)= ibarI(kE,1)+OE; %integration of the spiking input form the E pop in a short time window defined bu isiI
    
    
    
    
    
    [mE,kE]= max(VE - ThreshE-0.02*randn(NneuronE,1)); %choosing the neuron that has the biggest membrane potential
    
    
    
    if (mE>=0 && LSE(1,kE)<0 ) %if the voltage of this neuron is greater than the threshold and his refratory period is over it spikes
        
        OE=1;   % the spike variable in the population is turned to one
        LSE(1,kE)=10;   % the refractory period of this neuron is set to 10 time steps
        
        
        FE(:,kE)=FE(:,kE)+ epsf*(ibarE-1*FE(:,kE)); %the XE weights are updated using the FF rule
        CEE(:,kE)=max(CEE(:,kE) -epsr*(1*(VE+mu*rOE)+CEE(:,kE)),0);      %the EE weights are updated using the Recurr rule
        CEE(kE,kE)=-0.02; %the autpases are refixed
        
    else
        OE=0;
    end
    
    LSE=LSE-1; %on each time step the refractory variables are decreased
    
    VI=(1-lambda*dt)*VI+OE*CEI(:,kE)+OI*CII(:,kI)+0.002*randn(NneuronI,1); %the Voltages of the I neurons integrate the spikes from the E and I neurons
    
    
    [mI,kI]=max(VI - ThreshI-0.02*randn(NneuronI,1)); %choosing the neuron that has the biggest membrane potential
    
    
    
    
    if (mI>=0 && LSI(1,kI)<0) %if the voltage of this neuron is greater than the threshold and his refratory period is over it spikes
        
        OI=1;          % the spike variable in the population is turned to one
        LSI(1,kI)=10;  % the refractory period of this neuron is set to 10 time steps
        
        
        CEI(kI,:)=max(CEI(kI,:)+epsf*((ibarI)'-(CEI(kI,:))),0);  %the EI weights are updated using the FF rule
        CII(:,kI)=min(CII(:,kI) -epsr*(1*(VI+mu*rOI)+1*CII(:,kI)+mu*Id(:,kI)),0);       %the II weights are updated using the recurrent rule
        CIE(:,kI)=min(CIE(:,kI) -epsr*(1*(VE+ mu*rOE)+1*CIE(:,kI)),0);       %the IE weights are updated using the recurrent rule
        
    else
        
        OI=0;
    end
    
    rOI(kI,1)=rOI(kI,1)+OI;
    rOE(kE,1)=rOE(kE,1)+OE;
    
    rOE=(1-lambda*dt)*rOE;
    rOI=(1-lambda*dt)*rOI;
    LSI=LSI-1; %on each time step the refractory variables are decreased
    
    
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%   Computing Optimal Decoders  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%
%%%%% After having learned the connectivities we compute the
%%%%% optimal decoding weights for each instance of the network defined by
%%%%% FF and recurr connectivitiy matrices registered
%%%%% previously. This will allow us to compute the
%%%%% decoding error over learning.
%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Computing optimal decoders\n')
TimeL=40000;
DecsE=zeros(T,Nx,NneuronE);
DecsI=zeros(T,Nx,NneuronI);
InputL=A(1,1)*(mvnrnd(zeros(1,Nx),eye(Nx),TimeL))';

xL=zeros(Nx,TimeL);


for k=1:Nx
    InputL(k,:)=conv(InputL(k,:),w,'same');
end

for t=2:TimeL
    
    xL(:,t)= (1-lambda*dt)*xL(:,t-1)+ dt*InputL(:,t-1);
    
end

for i=1:T
    
    [rOEL, ~,~, rOIL, ~,~] = runnet(dt,TimeL, lambda,NneuronE,NneuronI,ThreshE,ThreshI,InputL,squeeze(FsE(i,:,:)),squeeze(CsEE(i,:,:)),squeeze(CsEI(i,:,:)),squeeze(CsII(i,:,:)),squeeze(CsIE(i,:,:)));
    DecsE(i,:,:)=(rOEL'\xL')';
    DecsI(i,:,:)=(rOIL'\xL')';
    
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%  Computing Decoding Error, rates through Learning %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%
%%%%% In this part we run the different instances of the network using a
%%%%% new test input and we measure the evolution of the dedocding error
%%%%% through learning using the decoders that we computed preciously. We also
%%%%% measure the evolution of the mean firing rate anf the variance of the
%%%%% membrane potential.
%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Computing decoding errors and rates over learning\n')
TimeT=10000; % size of the test input
MeanPrateE=zeros(1,T);%array of the mean rates over learning
MeanPrateI=zeros(1,T);%array of the mean rates over learning
ErrorE=zeros(1,T);%array of the decoding error over learning
ErrorI=zeros(1,T);%array of the decoding error over learning
MembraneVarE=zeros(1,T);%mean membrane potential variance over learning
MembraneVarI=zeros(1,T);%mean membrane potential variance over learning
xT=zeros(Nx,TimeT);%target ouput



Trials=10; %number of trials

for r=1:Trials %for each trial
    InputT=A*(mvnrnd(zeros(1,Nx),eye(Nx),TimeT))'; % we genrate a new input
    
    for k=1:Nx
        InputT(k,:)=conv(InputT(k,:),w,'same'); % we wmooth it
    end
    
    for t=2:TimeT
        xT(:,t)= (1-lambda*dt)*xT(:,t-1)+ dt*InputT(:,t-1); % ans we comput the target output by leaky inegration of the input
    end
    
    for i=1:T %for each instance of the network
        
        [ rOET, OET,VET, rOIT, OIT,VIT] = runnet(dt,TimeT, lambda,NneuronE,NneuronI,ThreshE,ThreshI,InputT,squeeze(FsE(i,:,:)),squeeze(CsEE(i,:,:)),squeeze(CsEI(i,:,:)),squeeze(CsII(i,:,:)),squeeze(CsIE(i,:,:)));%we run the network with current input InputT
        
        xestE=squeeze(DecsE(i,:,:))*rOET; %we deocode the ouptut using the optinal decoders previously computed
        xestI=squeeze(DecsI(i,:,:))*rOIT; %we deocode the ouptut using the optinal decoders previously computed
        
        if (i==1 && r==Trials)
            
            xestEB=xestE;
            xestIB=xestI;
            OEB=OET;
            OIB=OIT;
            
        end
        ErrorE(1,i)=ErrorE(1,i)+sum(var(xT-xestE,0,2))/(sum(var(xT,0,2))*Trials);%we comput the variance of the error normalized by the variance of the target E population
        ErrorI(1,i)=ErrorI(1,i)+sum(var(xT-xestI,0,2))/(sum(var(xT,0,2))*Trials);%we comput the variance of the error normalized by the variance of the target for I population
        MeanPrateE(1,i)=MeanPrateE(1,i)+sum(sum(OET))/(TimeT*dt*NneuronE*Trials);%we comput the average firing rate per neuron E population
        MeanPrateI(1,i)=MeanPrateI(1,i)+sum(sum(OIT))/(TimeT*dt*NneuronI*Trials);%we comput the average firing rate per neuron for I population
        MembraneVarE(1,i)=MembraneVarE(1,i)+sum(var(VET,0,2))/(NneuronE*Trials);% we compute the average membrane potential variance per neuron for E population
        MembraneVarI(1,i)=MembraneVarI(1,i)+sum(var(VIT,0,2))/(NneuronI*Trials);% we compute the average membrane potential variance per neuron for I population
    end
    
    
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%  Plotting Decoding Error, rates through Learning  %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

orange = [1 0.5 0.2];


figure
set(gcf,'Units','centimeters')
xSize = 50;  ySize =34;
xLeft = (21-xSize)/4; yTop = (30-ySize)/4;
set(gcf,'Position',[xLeft yTop xSize ySize]); %centers on A4 paper
set(gcf, 'Color', 'w');

lines=6;
fsize=11;

h=subplot(lines,3,[1 4]);
ax=get(h,'Position');
loglog((2.^(1:T(1,1)))*dt,ErrorE,'g');
hold on
loglog((2.^(1:T(1,1)))*dt,ErrorI,'Color',orange);
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
ylabel('Decoding Error')
title('Evolution Through Learning')
set(gca,'ticklength',[0.01 0.01]/ax(3))
box off


h=subplot(lines,3,[7 10]);
ax=get(h,'Position');
loglog((2.^(1:T(1,1)))*dt,MeanPrateE,'g');
hold on
loglog((2.^(1:T(1,1)))*dt,MeanPrateI,'Color',orange);
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
ylabel('Mean rate per neuron')
set(gca,'ticklength',[0.01 0.01]/ax(3))
box off


h=subplot(lines,3,[13 16]);
ax=get(h,'Position');
loglog((2.^(1:T(1,1)))*dt,MembraneVarE,'g');
hold on
loglog((2.^(1:T(1,1)))*dt,MembraneVarI,'Color',orange);
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
xlabel('time')
ylabel('mean M.P varaince per neuron')
set(gca,'ticklength',[0.01 0.01]/ax(3))
box off


h=subplot(lines,3,2);
plot(xestEB(1,:),'g');
hold on
plot(xestIB(1,:),'Color',orange);
plot(xT(1,:),'b')
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
ylabel('x,x_hat Dim1')
title('Before Learning')
box off


h=subplot(lines,3,3);
plot(xestE(1,:),'g');
hold on
plot(xestI(1,:),'Color',orange);
plot(xT(1,:),'b')
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
title('After Learning')
box off

h=subplot(lines,3,5);
plot(xestEB(2,:),'g');
hold on
plot(xestIB(2,:),'Color',orange);
plot(xT(2,:),'b')
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
ylabel('x,x_hat Dim2')
box off

h=subplot(lines,3,6);
plot(xestE(2,:),'g');
hold on
plot(xestI(2,:),'Color',orange);
plot(xT(2,:),'b')
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
xlabel('time')
box off

h=subplot(lines,3,8);
plot(xestEB(3,:),'g');
hold on
plot(xestIB(3,:),'Color',orange);
plot(xT(3,:),'b')
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
ylabel('x,x_hat Dim2')
box off

h=subplot(lines,3,9);
plot(xestE(3,:),'g');
hold on
plot(xestI(3,:),'Color',orange);
plot(xT(3,:),'b')
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
xlabel('time')
box off
%
h=subplot(lines,3,[11 14 17]);
plot(OEB.*(ones(NneuronE,1)*(1:TimeT)),OEB.*((1:NneuronE)'*ones(1,TimeT)),'.g');
hold on
plot(OIB.*(ones(NneuronI,1)*(1:TimeT)),OIB.*(((NneuronE+1):(NneuronE+NneuronI))'*ones(1,TimeT)),'.','Color',orange);
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
xlabel('time')
box off

h=subplot(lines,3,[12 15 18]);
plot(OET.*(ones(NneuronE,1)*(1:TimeT)),OET.*((1:NneuronE)'*ones(1,TimeT)),'.g');
hold on
plot(OIT.*(ones(NneuronI,1)*(1:TimeT)),OIT.*(((NneuronE+1):(NneuronE+NneuronI))'*ones(1,TimeT)),'.','Color',orange);
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
xlabel('time')
box off



end

