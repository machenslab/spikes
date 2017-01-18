function gb=gabor_fn(sigma,theta,lambda,psi,gamma,xcenter,ycenter)
 
% returns a gabor function

sigma_x = sigma;
sigma_y = sigma/gamma;
 
% Bounding box
boxsize=12-1;
xmax = 1;
ymax = 1;
xmin = -xmax ;
ymin = -ymax;
offsetx=xcenter*max(xmax,ymax);
offsety=ycenter*max(xmax,ymax);
X=xmin:((xmax-xmin)/boxsize):xmax;
Y=ymin:((ymax-ymin)/boxsize):ymax;
[x,y] = meshgrid(X,Y);
 
% Rotation 
x_theta=(x- offsetx)*cos(theta)+(y- offsety)*sin(theta);
y_theta=-(x- offsetx)*sin(theta)+(y- offsety)*cos(theta);
 
gb= exp(-.5*(x_theta.^2/sigma_x^2+y_theta.^2/sigma_y^2)).*cos(2*pi/lambda*x_theta+psi);
% gb=gb/max(max(gb));
end