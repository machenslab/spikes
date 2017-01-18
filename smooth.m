% [S, sks] = SMOOTH(x, sigma, [binout=1])
%
% sigma is the std of the smoothing gaussian, in bins. Each bin is
% one vector element in x. Wraparound is dealt with by padding the
% signal with "extensions" by a length of 3*sigma. E.g. if the
% signal is [1 0 0 -1], and sigma is 2/3, then the signal will be
% padded into [1 1 1 0 0 -1 -1 -1].
%
% If X is a matrix, each row is smoothed *except* if X is a single
% column vector, in which case that is smoothed.
%
% binout must be an integer >= 1. This is the subsampling rate for
% the output. Note that this is subsampling, not binning (sum will
% not be conserved). 
%
% Done in a dumb way here-- smoother prep should be directly in
% fourier space, not x-space.
%
% sks is the sum of the squared values of the smoothing kernel (in
% space). This is such that if the original signal had RMS 1, then
% the smoothed signal will have RMS = sqrt(sks).
%
% (c) Carlos Brody

function [S, sks] = smooth(x, sigma, binout)
    
   if nargin < 3, binout=1; end;
    
   wascol = 0;
   if size(x,2) == 1, x = x'; wascol = 1; end;
   
   extbins = ceil(3*sigma);
   orig_cols = size(x,2);
   x = [ones(size(x,1),1).*x(:,1)*ones(1,extbins), ...
	  x, ...
          ones(size(x,1),1).*x(:,end)*ones(1,extbins)];
   
   olength = size(x,2);
   ulength = 2.^nextpow2(olength);
   
   smoother = [0:(ceil(ulength/2) - 1), (-floor(ulength/2)):-1];
   smoother = exp(-smoother.*smoother./(2*sigma*sigma));
   smoother = smoother/sum(smoother);
   sks = sum(smoother.^2);
   smoother = fft(smoother);
 
   S = zeros(size(x,1), floor(orig_cols/binout));
   
   for r=1:size(x,1),
      
      X = [x(r,:), zeros(1, ulength - olength)];
      s = real(ifft(fft(X).*smoother));

      s = s(extbins+1:extbins+orig_cols);

      s = s(1:(binout*floor(orig_cols/binout)));
      s = reshape(s, binout, length(s)/binout);
      
      S(r,:) = s(1,:);
   end;
   
   if wascol,
      S = S';
   end;
   
   