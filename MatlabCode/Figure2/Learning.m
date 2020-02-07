function [Fs,Cs,F,C,Decs, ErrorC]=Learning(dt,lambda,epsr,epsf,alpha, beta, mu, Nneuron,Nx, Thresh,F,C)

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
%%%%  population Nneuron, the dimension of the input, the threshold of
%%%%  the neurons  an the initial feedforward and recuurrent connectivity F
%%%%  and C.
%%%%
%%%%   The output of this function are arrays, Fs abd Cs, containning the
%%%%   connectivity matrices sampled at exponential time instances Fs and
%%%%   Cs , The Final connectivity matrices F and C. It also gives the
%%%%   Optimal decoders for each couple of recurrent and feedforward
%%%%   connectivities registered in Fs and Cs. The output ErrorC contains
%%%%   the distance between the current and optimal recurrent connectivity
%%%%   stored in Cs. 
%%%%
%%%%   It also produces two figures. The first one it repsents the
%%%%   connectivities before and after learning and the second figure
%%%%   represents the performance of the network through learning. 
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

Nit=14000;   %number of iteration
Ntime=1000; %size of an input sequence
TotTime=Nit*Ntime;%total time of Learning

T=floor(log(TotTime)/log(2)); %Computing the size of the matrix where the weights are stocked on times defined on an exponential scale 
Cs=zeros(T,Nneuron, Nneuron); %the array that contains the different instances of reccurent connectivty through learning
Fs=zeros(T,Nx, Nneuron);      %the array that contains the different instances of feedforward connectivty through learning

V=zeros(Nneuron,1); %voltage vector of the population
O=0;  %variable indicating the eventual  firing of a spike
k=1;  %index of the neuron that fired
rO=zeros(Nneuron,1); %vector of filtered spike train

x=zeros(Nx,1);   %filtered input
Input=zeros(Nx,Ntime); %raw input to the network
Id=eye(Nneuron); %identity matrix

A=2000; %Amplitude of the input
sigma=abs(30); %std of the smoothing kernel
w=(1/(sigma*sqrt(2*pi)))* exp(-(([1:1000]-500).^2)/(2*sigma.^2));%gaussian smoothing kernel used to smooth the input
w=w/sum(w); % normalization oof the kernel


j=1; % index of the (2^j)-time step (exponential times)
l=1;

fprintf('%d percent of the learning  completed\n',0)
 
for i=2:TotTime
    
    if ((i/TotTime)>(l/100))
             fprintf('%d percent of the learning  completed\n',l)
        l=l+1;
    end
    
    if (mod(i,2^j)==0) %registering ther weights on an exponential time scale 2^j
        Cs(j,:,:)=C;   %registering the recurrent weights
        Fs(j,:,:)=F;   %registering the Feedfoward weights
        j=j+1;
    end
    
    if (mod(i-2,Ntime)==0) %Generating a new iput sequence every Ntime time steps 
        Input=(mvnrnd(zeros(1,Nx),eye(Nx),Ntime))'; %generating a new sequence of input which a gaussion vector
        for d=1:Nx
            Input(d,:)=A*conv(Input(d,:),w,'same'); %smoothing the previously generated white noise with the gaussian window w
        end     
    end
    
    V=(1-lambda*dt)*V + dt*F'*Input(:,mod(i,Ntime)+1)+ O*C(:,k)+0.001*randn(Nneuron,1); %the membrane potential is a leaky integration of the feedforward input and the spikes
    x=(1-lambda*dt)*x+dt*Input(:,mod(i,Ntime)+1); %filtered input
         
    [m,k]= max(V - Thresh-0.01*randn(Nneuron,1)-0); %finding the neuron with largest membrane potential
    
    
    if (m>=0) %if its membrane potential exceeds the threshold the neuron k spikes  
        O=1; % the spike ariable is turned to one
        F(:,k)=F(:,k)+epsf*(alpha*x-F(:,k)); %updating the feedforward weights
        C(:,k)=C(:,k) -(epsr)*(beta*(V+ mu*rO)+C(:,k)+mu*Id(:,k));%updating the recurrent weights
        rO(k,1)=rO(k,1)+1; %updating the filtered spike train
    else
        O=0;
    end
    
    rO=(1-lambda*dt)*rO; %filtering the spikes
       
end


fprintf('Learning  completed\n')

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%   Computing Optimal Decoders  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%
%%%%% After having learned the connectivities F and C we compute the
%%%%% optimal decoding weights for each instance of the network defined by
%%%%% the pairs of the FF and recurr connectivitiy matrices stocked
%%%%% previously in arrays  Fs and Cs. This will allow us to compute the
%%%%% decoding error over learning.
%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Computing optimal decoders\n')
TimeL=50000; % size of the sequence  of the input that will be fed to neuron
xL=zeros(Nx,TimeL); % the target output/input
Decs=zeros(T,Nx,Nneuron);% array where the decoding weights for each instance of the network will be stocked
InputL=0.3*A*(mvnrnd(zeros(1,Nx),eye(Nx),TimeL))'; %generating a new input sequence

for k=1:Nx
    InputL(k,:)=conv(InputL(k,:),w,'same'); %smoothing the input as before
end

for t=2:TimeL
    
    xL(:,t)= (1-lambda*dt)*xL(:,t-1)+ dt*InputL(:,t-1); %compute the target output by a leaky integration of the input
    
end


for i=1:T
    [rOL, ~, ~] = runnet(dt, lambda, squeeze(Fs(i,:,:)) ,InputL, squeeze(Cs(i,:,:)),Nneuron,TimeL, Thresh); % running the network with the previously generated input for the i-th instanc eof the network
    Dec=(rOL'\xL')'; % computing the optimal decoder that solves xL=Dec*rOL
    Decs(i,:,:)=Dec; % stocking the decoder in Decs
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
MeanPrate=zeros(1,T);%array of the mean rates over learning
Error=zeros(1,T);%array of the decoding error over learning
MembraneVar=zeros(1,T);%mean membrane potential variance over learning
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
        [rOT, OT, VT] = runnet(dt, lambda, squeeze(Fs(i,:,:)) ,InputT, squeeze(Cs(i,:,:)),Nneuron,TimeT, Thresh);%we run the network with current input InputL
        
        xestc=squeeze(Decs(i,:,:))*rOT; %we deocode the ouptut using the optinal decoders previously computed
        Error(1,i)=Error(1,i)+sum(var(xT-xestc,0,2))/(sum(var(xT,0,2))*Trials);%we comput the variance of the error normalized by the variance of the target
        MeanPrate(1,i)=MeanPrate(1,i)+sum(sum(OT))/(TimeT*dt*Nneuron*Trials);%we comput the average firing rate per neuron
        MembraneVar(1,i)=MembraneVar(1,i)+sum(var(VT,0,2))/(Nneuron*Trials);% we compute the average membrane potential variance per neuron     
    end
    
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%  Plotting Decoding Error, rates through Learning  %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
set(gcf,'Units','centimeters')
xSize = 24;  ySize =34;
xLeft = (21-xSize)/4; yTop = (30-ySize)/4;
set(gcf,'Position',[xLeft yTop xSize ySize]); %centers on A4 paper
set(gcf, 'Color', 'w');

lines=3;
fsize=11; 

%plotting the error
h=subplot(lines,1,1);
ax=get(h,'Position');
loglog((2.^(1:T(1,1)))*dt,Error,'k');
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
xlabel('time')
ylabel('Decoding Error')
title('Evolution of the Decoding Error Through Learning')
set(gca,'ticklength',[0.01 0.01]/ax(3))
box off

%plotting the mean rate
h=subplot(lines,1,2);
ax=get(h,'Position');
loglog((2.^(1:T(1,1)))*dt,MeanPrate,'k');
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
xlabel('time')
ylabel('Mean Rate per neuron')
title('Evolution of the Mean Population Firing Rate Through Learning ')
set(gca,'ticklength',[0.01 0.01]/ax(3))
box off

%plotting the mean membrane variance
h=subplot(lines,1,3);
ax=get(h,'Position');
loglog((2.^(1:T(1,1)))*dt,MembraneVar,'k');
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
xlabel('time')
ylabel('Voltage Variance per Neuron')
title('Evolution of the Variance of the Membrane Potential ')
set(gca,'ticklength',[0.01 0.01]/ax(3))
box off



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%   Computing distance to  Optimal weights through Learning %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% 
%%%%%% we compute the distance between the recurrent connectivity matrics
%%%%%% ,stocked in Cs, and FF^T through learning.
%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


ErrorC=zeros(1,T);%array of distance between connectivity

for i=1:T %for each instance od the network
    
 CurrF=squeeze(Fs(i,:,:)); 
 CurrC=squeeze(Cs(i,:,:)); 
    
    
Copt=-CurrF'*CurrF; % we comput FF^T
optscale = trace(CurrC'*Copt)/sum(sum(Copt.^2)); %scaling factor between the current and optimal connectivities
Cnorm = sum(sum((CurrC).^2)); %norm of the actual connectivity
ErrorC(1,i)=sum(sum((CurrC - optscale*Copt).^2))/Cnorm ;%normalized error between the current and optimal connectivity

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Plotting Weights  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



figure

set(gcf,'Units','centimeters')
xSize = 24;  ySize =34;
xLeft = (21-xSize)/4; yTop = (30-ySize)/4;
set(gcf,'Position',[xLeft yTop xSize ySize]); %centers on A4 paper
set(gcf, 'Color', 'w');



lines=4;
fsize=11; 


%Plotting the evolution of distance between the recurrent weights and FF^T through learning
h=subplot(lines,2,[1 2]);
ax=get(h,'Position');
loglog((2.^(1:T))*dt,ErrorC,'k');
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
xlabel('time')
ylabel('Distance to optimal weights')
title('Weight Convergence')
set(gca,'ticklength',[0.01 0.01]/ax(3))

%ploting the feedforward weighs in a 2D plane before learning
h=subplot(lines,2,3);
ax=get(h,'Position');
Fi=squeeze(Fs(1,:,:)); 
plot(Fi(1,:),Fi(2,:),'.k');
hold on
plot(0,0,'+');
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
set(gca,'ticklength',[0.01 0.01]/ax(3))
axis([-1, 1 -1 1]);
xlabel('FF Weights Component 1')
ylabel('FF Weights Component 2')
title('Before Learning')
axis square

%ploting the feedforward weighs in a 2D plane After learning
h=subplot(lines,2,4);
ax=get(h,'Position');
plot(F(1,:),F(2,:),'.k');
hold on
plot(0,0,'+');
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
set(gca,'ticklength',[0.01 0.01]/ax(3))
axis([-1, 1 -1 1]);
xlabel('FF Weights Component 1')
ylabel('FF Weights Component 2')
title('After Learning')
axis square

%scatter plot of C and FF^T before learning
h=subplot(lines,2,5);
ax=get(h,'Position');
Ci=squeeze(Cs(1,:,:)); 
plot(Ci,-Fi'*Fi,'.k');
hold on
plot(0,0,'+');
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
set(gca,'ticklength',[0.01 0.01]/ax(3))
axis([-1, 1 -1 1]);
xlabel('FF^T')
ylabel('Learned Rec Weights')
axis square


%scatter plot of C and FF^T After learning
h=subplot(lines,2,6);
ax=get(h,'Position');
plot(C,-F'*F,'.k');
hold on
plot(0,0,'+');
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
set(gca,'ticklength',[0.01 0.01]/ax(3))
axis([-1, 1 -1 1]);
xlabel('FF^T')
ylabel('Learned Rec Weights')
axis square

%scatter plot of optimal decoder and F^T before learning
h=subplot(lines,2,7);
ax=get(h,'Position');
plot(squeeze(Decs(1,:,:)),Fi,'.k');
hold on
plot(0,0,'+');
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
set(gca,'ticklength',[0.01 0.01]/ax(3))
axis([-1, 1 -1 1]);
xlabel('Optimal decoder')
ylabel('F^T')
axis square

%scatter plot of optimal decoder and F^T After learning
h=subplot(lines,2,8);
plot(squeeze(Decs(T,:,:)),F,'.k');
hold on
plot(0,0,'+');
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
set(gca,'ticklength',[0.01 0.01]/ax(3))
axis([-1, 1 -1 1]);
xlabel('Optimal decoder')
ylabel('F^T')
axis square

end











