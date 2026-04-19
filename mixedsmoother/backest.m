function [xnew, signewsq, a] = backest(x, xold, sigsq, sigsqold);
%BACKEST  Backward smoothing filter for mixed-type learning-state model
%
%   Usage:
%       [xnew, signewsq, a] = backest(x, xold, sigsq, sigsqold)
%
%   Inputs:
%       x        : 1xT double - posterior mode x{k|k} -- required
%       xold     : 1xT double - one-step prediction x{k|k-1} -- required
%       sigsq    : 1xT double - posterior variance SIG^2{k|k} -- required
%       sigsqold : 1xT double - one-step prediction variance SIG^2{k|k-1} -- required
%
%   Outputs:
%       xnew     : 1xT double - backward (smoothed) state estimate x{k|K}
%       signewsq : 1xT double - backward state variance SIG^2{k|K}
%       a        : 1xT double - smoother gain A{k} = sigsq(k)/sigsqold(k+1)
%
%   See also: recfilter, m_step, mixedlearningcurve
%
%   ∿∿∿  Prerau Laboratory MATLAB Codebase · sleepEEG.org  ∿∿∿
%        Source: https://github.com/preraulab/labcode_main

T = size(x,2);
a=0;
xnew(T)     = x(T);
signewsq(T) = sigsq(T);
for i = T-1 :-1: 2
   a(i)        = sigsq(i)/sigsqold(i+1);
   xnew(i)     = x(i) + a(i)*(xnew(i+1) - xold(i+1));
   signewsq(i) = sigsq(i) + a(i)*a(i)*(signewsq(i+1)-sigsqold(i+1));
end


