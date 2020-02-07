Nneuron=20; % size of the population
Nx=2;       %dimesnion of the input

lambda=50;    %membrane leak
dt=0.001;     %time step

epsr=0.001;  % earning rate of the recurrent connections
epsf=0.0001; %% learning rate of the feedforward connections FF

alpha=0.18; % scaling of the Feefforward weights
beta=1/0.9;  %scaling of the recurrent weights
mu=0.02/0.9; %quadratic cost


%%Initial connectivity

Fi=0.5*randn(Nx,Nneuron); %the inital feedforward weights are chosen randomely
Fi=1*(Fi./(sqrt(ones(Nx,1)*(sum(Fi.^2)))));%the FF weights are normalized
Ci=-0.2*(rand(Nneuron,Nneuron))-0.5*eye(Nneuron); %the initial recurrent conectivity is very weak except for the autapses

Thresh=0.5; %vector of thresholds of the neurons


[Fs,Cs,F,C,Decs,ErrorC]=Learning(dt,lambda,epsr,epsf,alpha, beta, mu, Nneuron,Nx, Thresh,Fi,Ci);
