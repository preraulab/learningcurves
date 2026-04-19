function [xnew, signewsq, A] = backwardfilter(x, xold, sigsq, sigsqold);
%BACKWARDFILTER  Backward (RTS) smoother for the binary learning-state model
%
%   Usage:
%       [xnew, signewsq, A] = backwardfilter(x, xold, sigsq, sigsqold)
%
%   Inputs:
%       x        : 1xT double - posterior mode x{k|k} -- required
%       xold     : 1xT double - one-step prediction x{k|k-1} -- required
%       sigsq    : 1xT double - posterior variance SIG^2{k|k} -- required
%       sigsqold : 1xT double - one-step prediction variance SIG^2{k|k-1} -- required
%
%   Outputs:
%       xnew     : 1xT double - backward estimate of learning state x{k|K} (equation A.10)
%       signewsq : 1xT double - backward estimate of state variance SIG^2{k|K} (equation A.12)
%       A        : 1xT double - smoother gain A{k} = sigsq(k)/sigsqold(k+1) (equation A.11)
%
%   Notes:
%       Equation references follow Smith et al., J Neurosci 2004 (Appendix A).
%
%   See also: forwardfilter, em_bino, binsmoother
%
%   ∿∿∿  Prerau Laboratory MATLAB Codebase · sleepEEG.org  ∿∿∿

T = size(x,2);

%Initial conditions: use values of posterior mode and posterior variance
xnew(T)     = x(T);
signewsq(T) = sigsq(T);


for i = T-1 :-1: 2
 %for each posterior mode prediction, compute new estimates given all of
 %the data from the experiment (estimates from ideal observer)
   A(i)        = sigsq(i)/sigsqold(i+1);
   xnew(i)     = x(i) + A(i)*(xnew(i+1) - xold(i+1));
   signewsq(i) = sigsq(i) + A(i)*A(i)*(signewsq(i+1)-sigsqold(i+1));
end
