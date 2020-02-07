


NneuronE=300; %size of the E population
NneuronI=75; %size of the I population
Nx=3;        %dimension of the input
lambda=50;   %leak of the membrane potential
dt=0.0001;   %time step

isiE=lambda*6; %integration constant of the feedforward input to the E population used in the XE plastcity rule
isiI=lambda*1; %integration constant of the spiking input to the I population fom the E population used in the EI plasticity rule


epsf=0.00001;   %learning rate for the  learning rule for XE and EI plasticity rules
epsr=0.0001;    %learning rate for the  learning rule for EE, II and EI plasticity rules

mu=0.002;

Nit=5 ;  %number of iterations
Ntime=1000; %number of steps in an iteration



FE=1*randn(Nx,NneuronE);   %Generation of radom feedforward weights to the E population
FE=1*(FE./(sqrt(ones(Nx,1)*(sum(FE.^2)))));%normalization of the input weights

ThreshE=((sum(FE.^2,1))/2)'; % thresholds for the E neurons are set to half of the norm of the FF weights


ThreshI=ThreshE(1,1)*ones(NneuronI,1); % threshold for I neuros are the same as for the E neurons

Id=eye(NneuronI);           %identity matrix
CEI=0.5*[Id Id Id Id];      % the E to I connections are initialized such as every I neuron is connected to 4 E neurons


CEE=-0.02*eye(NneuronE); % the EE conections are set to zeron except fot the weak autapses
CII=-0.5*eye(NneuronI);  % II neurons are same as EE neurons
CIE=-0.3*CEI';           % the IE to connectivity has the same initial structure as the EI connectivity


[FsE,CsEE,CsEI,CsII,CsIE,DecsE,DecsI,OEB,OIB,TimeT,T]=Learning(Nx,dt,lambda,epsr,epsf, mu, NneuronE,NneuronI, ThreshE,ThreshI,isiE,isiI, FE,CEE,CEI,CII,CIE);

FanoCV; % computes and plots the Fano Factor and CV through learning for the Excitatory population



