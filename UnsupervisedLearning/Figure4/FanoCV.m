Nt=10000; % time steps in each trial
NInputs=20; % number of different inputs for wich the fano factors and CVs are calculated
Ntrials=50; % number of trials for each input to compute the FF and the CV

S=size(FsE);
P=S(1);

Fanos=zeros(NInputs,P);
CVs=zeros(NInputs,P);
Fanosimple=zeros(NneuronE,NInputs);
CVsimple=zeros(NneuronE,NInputs);
Counts=zeros(NneuronE,50);
ISIall=zeros(NneuronE,800);

NbISI=zeros(NneuronE,1);


for h=1:NInputs
    
    U=randn(Nx,1); %Generating a random vector
    U=U./(sum(U.^2).^0.5);%normalizing the vector
    Input= 90*U*ones(1,Nt);% generating a constant input in time equal to the previous vector
    
    for r=1:T %for each instance of the network
      
        ISIall=zeros(NneuronE,1);
        NbISI=zeros(NneuronE,1);
        
        for  s=1:Ntrials
            
            
            %we run the network using the input for Ntrials time
            [rOEf, OEf, VEf,~, ~,~] = runnet(dt,Nt, lambda,NneuronE,NneuronI,ThreshE,ThreshI,Input,squeeze(FsE(r,:,:)),squeeze(CsEE(r,:,:)),squeeze(CsEI(r,:,:)),squeeze(CsII(r,:,:)),squeeze(CsIE(r,:,:)));
            
            Counts(:,s)=sum(OEf,2); %for each trial we register the number of spikes emitted bt each neuron
            
            
            
            
            
            for g=1:NneuronE
                U=diff(find(OEf(g,:)>0));
                ISIall(g,(NbISI(g)+1):(NbISI(g)+length(U)))=U; % we conpute the ISIs (inter-spike interval)
                NbISI(g)=NbISI(g)+length(U);  % and the number of spikes  per trial
            end
        end
        
               
        p=1;
        
        Fanosimple=zeros(NneuronE,1);
        CVsimple=zeros(NneuronE,1);
        
        for g=1:NneuronE
            
            if(mean(Counts(g,:))>1) % if the neuron g has more than one spike
                
                Fanosimple(g)=  (var(Counts(g,:)))./(mean(Counts(g,:)));  % we compute the fano factor over the Ntrial trials for this input  for each neuron
                p=p+1; % the number of neuron that has more than one spike is incremented
                CVsimple(g)= std(ISIall(g,1:NbISI(g)))./mean(ISIall(g,1:NbISI(g)));% we compute the CV over the Ntrial trials for this input  for each neuron
                
            end
        end
        
        Fanoall=sum(Fanosimple)/(p-1);% average the the Fano factors  and CVs over neurons that has more than one spike
        CVall=sum(CVsimple)/(p-1);
        
        Fanos(h,r)=Fanoall; % we register these values
        CVs(h,r)=CVall;
        
        
    end    
    
end

Fano=sum(Fanos,1)/NInputs;% we average the fano factors over the NInputs different constant inputs
CV=sum(CVs,1)/NInputs;


% plotting the fano factors and CVs

figure
set(gcf,'Units','centimeters')
xSize = 50;  ySize =34;
xLeft = (21-xSize)/4; yTop = (30-ySize)/4;
set(gcf,'Position',[xLeft yTop xSize ySize]); %centers on A4 paper
set(gcf, 'Color', 'w');

lines=2;
fsize=11;

h=subplot(lines,1,1);
ax=get(h,'Position');
semilogx((2.^(1:T(1,1)))*dt,Fano,'k')
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
title('Fano factor through learing')
set(gca,'ticklength',[0.01 0.01]/ax(3))
box off


h=subplot(lines,1,2);
ax=get(h,'Position');
semilogx((2.^(1:T(1,1)))*dt,CV,'k');
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
title('Coefficient of variation through Learning')
xlabel('Time')
set(gca,'ticklength',[0.01 0.01]/ax(3))
box off
