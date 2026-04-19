function  [p, xhat, sigsq, xhatold, sigsqold] ...
		= forwardfilter(I, sigE, xguess, sigsqguess, mu);
%FORWARDFILTER  Forward recursive filter for the binary learning-state model
%
%   Usage:
%       [p, xhat, sigsq, xhatold, sigsqold] = ...
%           forwardfilter(I, sigE, xguess, sigsqguess, mu)
%
%   Inputs:
%       I          : 2xK double - row1: number correct per trial; row2: max possible -- required
%       sigE       : double - state random-walk standard deviation SIG_EPSILON -- required
%       xguess     : double - initial state guess x(1) -- required
%       sigsqguess : double - initial state variance guess SIG^2(1) -- required
%       mu         : double - logit of background (chance) probability -- required
%
%   Outputs:
%       p        : 1xK+1 double - observation-model probability p{k|k} (equation 2.2)
%       xhat     : 1xK+1 double - posterior mode x{k|k} (equation A.8)
%       sigsq    : 1xK+1 double - posterior variance SIG^2{k|k} (equation A.9)
%       xhatold  : 1xK+1 double - one-step prediction x{k|k-1} (equation A.6)
%       sigsqold : 1xK+1 double - one-step prediction variance SIG^2{k|k-1} (equation A.7)
%
%   Notes:
%       Uses Newton's method (newtonsolve) for the nonlinear posterior mode;
%       prints a diagnostic listing any trial indices where Newton failed.
%       Equation references follow Smith et al., J Neurosci 2004.
%
%   See also: backwardfilter, newtonsolve, binsmoother
%
%   ∿∿∿  Prerau Laboratory MATLAB Codebase · sleepEEG.org  ∿∿∿
%        Source: https://github.com/preraulab/labcode_main

K = size(I,2);
N = I(1,:);
Nmax = I(2,:);

%Initial conditions: use values from previous iteration
xhat(1)   = xguess;
sigsq(1)  = sigsqguess;
number_fail = [];

for k=2:K+1
   %for each trial, compute estimates of the one-step prediction, the
   %posterior mode (using Newton's Method), and the posterior variance
   %(estimates from subject's POV)

   %Compute the one-step prediction estimate of mean and variance
   xhatold(k)  = xhat(k-1);
   sigsqold(k) = sigsq(k-1) + sigE^2;

   %Use Newton's Method to compute the nonlinear posterior mode estimate
   [xhat(k),flagfail] = newtonsolve(mu,  xhatold(k), sigsqold(k), N(k-1), Nmax(k-1));

   if flagfail>0 %if Newton's Method fails, number_fail saves the time step
      number_fail = [number_fail k];
   end

   %Compute the posterior variance estimate
   denom       = -1/sigsqold(k) - Nmax(k-1)*exp(mu)*exp(xhat(k))/(1+exp(mu)*exp(xhat(k)))^2;
   sigsq(k)    = -1/denom;

end

if isempty(number_fail)<1
   fprintf(2,'Newton convergence failed at times %d \n', number_fail)
end

%Compute the observation model probability estimate
p = exp(mu)*exp(xhat)./(1+exp(mu)*exp(xhat));

