function orplot( th, F, size1 )

%ORPLOT( TH, F, AX) generates a tuning curve plot
%in polar coordinates. TH is the angle, F the firing
%rate, and AX the radial axis limit

NF = length(F);
Fx = F.*cos(th');
Fy = F.*sin(th');

set(gca,'FontSize',25, 'LineWidth', 2)
hold on;
plot(Fx,Fy,'-k','MarkerSize', 15,'LineWidth', 1.5);
for j=1:NF
  plot([0, Fx(j)],[0,Fy(j)],'-k','MarkerSize', 15,'LineWidth', 1.5);
end
axis(1.25 * [-size1 size1 -size1 size1]);
plot(size1*cos(0:(pi/50):2*pi),size1*sin(0:(pi/50):2*pi),'-k','LineWidth', 1.5);
plot([0, 0],[size1, -size1],'-k','LineWidth', 1);
plot([size1, -size1],[0, 0],'-k','LineWidth', 1);
plot([0, size1],[size1*1.24, size1*1.24,],'-k','LineWidth', 1.5)
plot([0, 0],[size1*1.2, size1*1.24],'-k','LineWidth', 1.5)
plot([size1, size1],[size1*1.2, size1*1.24],'-k','LineWidth', 1.5)
text(size1*0.28,size1*1.1,[num2str(size1,3) ' Hz'],'FontSize',15)
text(size1*1.1,size1*0.025,'0^o','FontSize',15)
text(-size1*0.1,-size1*1.1,'90^o','FontSize',15);
axis off;
box off;
axis square;
