
clear Ecl Icl Ecln Icln
 
gain=0.03;

for i=1:Nneurontot
[Ocl,rOcl,Vcl,xcl]=netruncl(Is*gain,Wref,Gamma,Nneurontot,Ninput,thres,i);
E=Gamma*xcl;
I=-(Wref.*(1-diag(ones(1,Nneurontot))))*rOcl;
Ecl(i,:)=E(i,:);
Icl(i,:)=I(i,:);
end

for i=1:Nneurontot
[Ocl,rOcl,Vcl,xcl]=netruncl(Is*0.03,Wrefnew,Gammanew,Nneurontot,Ninput,thresnew,i);
E=Gammanew*xcl;
I=-(Wrefnew.*(1-diag(ones(1,Nneurontot))))*rOcl;
Ecln(i,:)=E(i,:);
Icln(i,:)=I(i,:);
end

for i=1:Nneurontot
[Ocl,rOcl,Vcl,xcl]=netruncl(Is*0.03,Wrefnl,Gammanl,Nneurontot,Ninput,thresnl,i);
E=Gammanl*xcl;
I=-(Wrefnl.*(1-diag(ones(1,Nneurontot))))*rOcl;
Eclnl(i,:)=E(i,:);
Iclnl(i,:)=I(i,:);
end

%figure(1)

%plot(mean(max(Ecl,0)+max(Icl,0)),'r')
%hold on
%plot(mean(min(Ecl,0)+min(Icl,0)),'k')
%plot(Ecl'+Icl','k')

%figure(2)

%plot(mean(max(Ecln,0)+max(Icln,0)),'r')
%hold on
%plot(mean(min(Ecln,0)+min(Icln,0)),'k')
%plot(Ecln'+Icln','k')



