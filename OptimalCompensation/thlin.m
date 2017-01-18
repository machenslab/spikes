% Y = THLIN( X )           Threshold-linear function
%
% Returns Y=X if X>0, otherwise returns Y=0.

function y = thlin(x)

y = x;
y(y<0) = 0;
