Nneurontot=100;
Ninput=25;
lambda=8;
dt=0.00005;
Nshow=16000;

Nlearn=100;


%tolerance for pinv(Gamma)
pr=0.1;

%gain of test stimulus
gain=0.03;  %Gain of speech signals
gains=0.03; %Gain of new stimulus (two tones)

%Load optimal parameters and compute decoder
load Icurrent;
Dsp=pinv(Gamma,pr)*Wref;

%Example cell for panels A,B,C 
irec=36;

%frequencies activated by the new stimulus (for panel D,E,F)
freq1=19;
freq2=21;

%load speech stimulus
load speech;
SPu=speech.data;

%Select an example speech signal
Un=randperm(230500);
Iex=zeros(1,49901);
    
for i=1:Ninput
        
        SPu(i,Un(1)+1:Un(1)+5)=0;        
        cI(i,:)=(SPu(i,Un(1)+1)*100+(cumsum(interp1([1:500],gradient(SPu(i,Un(1)+1:Un(1)+500)),[1:0.01:500]))))*0.1;
        Iex(i,:)=gradient(cI(i,:))+lambda*dt*cI(i,:);
end

Iex=Iex(:,1:20000);


%Select a longer speech signal to train the decoder of the initial network. 
Un=randperm(230500);
    
for i=1:Ninput
        
        SPu(i,Un(1)+1:Un(1)+5)=0;        
        cI(i,:)=(SPu(i,Un(1)+1)*100+(cumsum(interp1([1:500],gradient(SPu(i,Un(1)+1:Un(1)+500)),[1:0.01:500]))))*0.1;
        Iexbis(i,:)=gradient(cI(i,:))+lambda*dt*cI(i,:);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Network with random weights (initial network) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Gammar(:,1:Ninput)=randn(Nneurontot,Ninput)*0.025;
Wrefr=randn(Nneurontot,Nneurontot)*0.1+diag(ones(1,Nneurontot))*0.8;
thresr=0.5;

[Or,rOr,Vr,xr]=netrun(Iex*gain,Wrefr,Gammar,Nneurontot,Ninput,thresr); 

%estimate decoder
[Ort,rOrt,Vrt,xrt]=netrun(Iexbis*gain,Wrefr,Gammar,Nneurontot,Ninput,thresr);
Dr=xrt/rOrt;

    %plot results
    
    subplot(7,3,1:2)
    contourf(xr(:,1:Nshow))
    hold on
    contour(xr(:,1:Nshow))
    caxis([-1 max(max(xr))])
    hold off
    
    subplot(7,3,4:5)
    contourf((Dr*rOr(:,1:Nshow)))
    hold on
    contour((Dr*rOr(:,1:Nshow)))
    caxis([-1 max(max(xr))])
    hold off
    
    subplot(7,3,7:8)
    plot((Or(:,1:Nshow).*([1:Nneurontot]'*ones(1,length(Or(:,1:Nshow)))))','k.')
    axis([1 Nshow 2 Nneurontot]) 
    
    subplot(7,3,10:11)
    E=(Gammar*xr(:,1:Nshow));
    I=-(Wrefr.*(1-diag(ones(1,Nneurontot))))*rOr(:,1:Nshow);
    plot(max(E(irec,1:Nshow),0)+max(I(irec,1:Nshow),0),'r')
    hold on
    plot(min(I(irec,1:Nshow),0)+min(E(irec,1:Nshow),0),'b')
    plot(E(irec,1:Nshow)'+I(irec,1:Nshow)','k')
    axis([1 Nshow -8 8])
    hold off
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Network with optimal weights %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
%Run network
[Osp,rOsp,Vsp,xsp]=netrun(Iex*gain,Wref,Gamma,Nneurontot,Ninput,thres);

    %plot results
    
    subplot(7,3,13:14)
    contourf((Dsp*rOsp(:,1:Nshow)))
    hold on
    contour((Dsp*rOsp(:,1:Nshow)))
    caxis([-1 max(max(xsp))])
    
    hold off
    
    subplot(7,3,16:17)
    plot((Osp(:,1:Nshow).*([1:Nneurontot]'*ones(1,length(Osp(:,1:Nshow)))))','k.')
    axis([1 Nshow 2 Nneurontot])

    subplot(7,3,19:20)
    E=(Gamma*xsp);
    I=-Wref*rOsp;
    plot(max(E(irec,1:Nshow),0)+max(I(irec,1:Nshow),0),'r')
    hold on
    plot(min(I(irec,1:Nshow),0)+min(E(irec,1:Nshow),0),'b')
    plot(E(irec,1:Nshow)'+I(irec,1:Nshow)','k')
    axis([1 Nshow -25 25])
    hold off


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initial response to new stimulus %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%create new stimulus
s=zeros(25,7000);
s(freq1,2000:5000)=0.1;
s(freq2,2000:5000)=0.1;

clear si sii Is;

for i=1:Ninput
    %temporally smooth        
    sii(i,:)=conv2(s(i,:),conv2(exp(-[1:1000]/100),exp(-[1:1000]/300)),'same');
    si(i,:)=conv2(sii(i,:),conv2(exp(-[1:1000]/100),exp(-[1:1000]/300)),'same')/55000;
    Is(i,:)=gradient(si(i,:))+lambda*dt*si(i,:);
end

%run network
[Oinit,rOinit,Vinit,xinit]=netrun(Is*gains,Wref,Gamma,Nneurontot,Ninput,thres);


%plot results
   
    subplot(7,3,3)
    contourf(xinit)
    hold on
    contour(xinit)
    axis([1 7000 1 25])
    caxis([-1 45])
    hold off
    
    subplot(7,3,6)
    contourf((Dsp*rOinit))
    hold on
    contour((Dsp*rOinit))
    axis([1 7000 1 25])
    caxis([-1 45])
    hold off

    subplot(7,3,9)
    plot((Oinit.*([1:Nneurontot]'*ones(1,length(Oinit))))','k.')
    axis([1 7000 2 Nneurontot])

   
    


%%%%%%%%%%%%%%%%%%%%%%%%%    
% Learn the new feature %
%%%%%%%%%%%%%%%%%%%%%%%%%


%load if re-learnt weights available
if learnanew==0
    load Icurrentnew;
    load Icurrentnl;
end

%Re-learn otherwise

clear Il;
if learnanew==1
    
    Gammanew=Gamma;
    thresnew=thres;
    Uccnew=Ucc;
    Wrefnew=Wref;
    mInew=mI;
    
    Gammanl=Gamma;
    thresnl=thres;
    Uccnl=Ucc;
    Wrefnl=Wref;
    mInl=mI;

    for it=1:Nlearn
    
     %Generate training stimuli = alternate speech/new stimulus
    
        Un=randperm(230500);
        Sti=SPu(:,Un(1):Un(1)+500);
        Sti(:,1:5)=0;
        Sti(:,10:50)=0;
        Sti(:,110:150)=0;
        Sti(:,210:250)=0;
        Sti(:,310:350)=0;
        Sti(:,410:450)=0;  
    
        for i=1:Ninput
    
            if i==freq1 || i==freq2
            
                Sti(i,10:50)=50; 
                Sti(i,110:150)=50;       
                Sti(i,210:250)=50;
                Sti(i,310:350)=50;        
                Sti(i,410:450)=50;
        
            end
       
            cI(i,:)=(Sti(i,1)*100+(cumsum(interp1([1:500],gradient(Sti(i,1:500)),[1:0.01:500]))))*0.1;
            Il(i,:)=gradient(cI(i,:))+lambda*dt*cI(i,:);
            
        end
        
        %learn FF + Lat connections
        [Onew,~,Vnew,~,~,~,Wrefnew,Gammanew,thresnew,Uccnew,mInew]=netadapt(Il*gain,Wrefnew,Gammanew,Nneurontot,Ninput,thres,Uccnew,mInew,5,5);
        
        %learn only FF connections
        [Onewnl,~,Vnewnl,~,~,~,Wrefnl,Gammanl,thresnl,Uccnl,mInl]=netadapt(Il*gain,Wrefnl,Gammanl,Nneurontot,Ninput,thresnl,Uccnl,mInl,0.25,0);
        
        [it,sum(sum(Onew)),sum(Onew(irec,:)),sum(sum(Onewnl)),sum(Onewnl(irec,:))]  

    end
    save Icurrentnew Gammanew Wrefnew thresnew mI Uccnew
    save Icurrentnl Gammanl Wrefnl thresnl mI Uccnl
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test and plot learning result %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Run re-trained network on new stimulus
[Omed,rOmed,Vmed,xmed]=netrun(Is*gains,Wrefnew,Gammanew,Nneurontot,Ninput,thresnew);

%Compute the decoder of the re-trained network
Dmed=pinv(Gammanew,pr)*Wrefnew;
 
    %plot results
    
    subplot(7,3,15)
    contourf((Dmed*rOmed))
    hold on
    contour((Dmed*rOmed))
    axis([1 7000 1 25])
    caxis([-1 45])
    hold off
   
    subplot(7,3,18)
    plot((Omed.*([1:Nneurontot]'*ones(1,length(Omed))))','k.')
    axis([1 7000 2 Nneurontot])
    


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run the network with only FF trained %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[Onl,rOnl,Vnl,xnl]=netruncl(Is*gains,Wrefnl,Gammanl,Nneurontot,Ninput,thresnl,irec);

%use the new decoder
Dnl=pinv(Gammanl,pr)*Wrefnl;
 

% De-comment if you want to plot the results of the network learning only
% FF connections (supplementary fig 2): 
    %plot results
  %  figure(3)
  %  subplot(4,3,3)
  %  plot((Onl.*([1:Nneurontot]'*ones(1,length(Onl))))','.')
  %  axis([1 7000 1 Nneurontot])

  %  subplot(4,3,6)
  %  contourf(xnl)
  %  hold on
  %  contour(xnl)
  %  axis([1 7000 1 25])
  %  caxis([-1 45*gain])
  %  hold off
  %  subplot(4,3,9)
  %  U=(Dnl*rOnl);
  %  contourf((Dnl*rOnl))
  %  hold on
  %  contour((Dnl*rOnl))
  %  axis([1 7000 1 25])
  %  caxis([-1 45*gain])
  %  hold off
  
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Use simulated patch clamp to measure the E and I currents %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    clampall;
    
    % Before re-training
    
    subplot(7,3,12)
    plot(Ecl'+Icl','color',[0.7,0.7,0.7])
    hold on
    plot(mean(max(Ecl,0)+max(Icl,0)),'r')
    plot(mean(min(Ecl,0)+min(Icl,0)),'b')
    axis([1 7000 -25 25])
    hold off
    
    % After re-training
    
    
    subplot(7,3,21)
    plot(Ecln'+Icln','color',[0.7,0.7,0.7])    
    hold on
    plot(mean(max(Ecln,0)+max(Icln,0)),'r')
    plot(mean(min(Ecln,0)+min(Icln,0)),'b')
    axis([1 7000 -25 25])
    hold off
    
    
    % de-comment if you want to plot the results for the network with only
    % FF plasticity (supplementary fig 2). 
    
   %figure (3)
   % subplot(4,3,12)
   % plot(mean(max(Eclnl,0)+max(Iclnl,0)),'r')
   % hold on
   % plot(mean(min(Eclnl,0)+min(Iclnl,0)),'b')
   % plot(Eclnl'+Iclnl','k')
   % hold off
