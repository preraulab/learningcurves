function  [xhat, sigsq, xhatold, sigsqold] ...
    = recfilter(N, Z, sig2e, sig2v, xguess, muone, rho, beta, alpha, gamma)
%RECFILTER  Forward recursive filter for the mixed binary/continuous learning model
%
%   Usage:
%       [xhat, sigsq, xhatold, sigsqold] = recfilter(N, Z, sig2e, sig2v, ...
%               xguess, muone, rho, beta, alpha, gamma)
%
%   Inputs:
%       N      : 1xT double - binary/count observations (number correct per trial) -- required
%       Z      : 1xT double - continuous observations (e.g. reaction time) -- required
%       sig2e  : double - RT observation noise variance -- required
%       sig2v  : double - state (random-walk) variance -- required
%       xguess : double - initial state guess x(1) -- required
%       muone  : double - logit of background (chance) probability -- required
%       rho    : double - state AR(1) coefficient -- required
%       beta   : double - RT observation slope -- required
%       alpha  : double - RT observation intercept -- required
%       gamma  : double - binary observation state weight -- required
%
%   Outputs:
%       xhat     : 1xT+1 double - posterior mode x{k|k}
%       sigsq    : 1xT+1 double - posterior variance SIG^2{k|k}
%       xhatold  : 1xT+1 double - one-step prediction x{k|k-1}
%       sigsqold : 1xT+1 double - one-step prediction variance SIG^2{k|k-1}
%
%   Notes:
%       Prints a diagnostic and returns zeros if x_newtonsolve fails to converge at
%       any time step.
%
%   See also: backest, x_newtonsolve, mixedlearningcurve
%
%   ∿∿∿  Prerau Laboratory MATLAB Codebase · sleepEEG.org  ∿∿∿

failed=0;

T = length(N);
cornum = N;


%set up some initial values
xhat(1) = xguess;
sigsq(1) = sig2v;

count = 1;

%number_fail saves the time steps if Newton method fails
number_fail = [];

%loop through all time
for t=2:T+1
    xhatold(t)  = rho*xhat(t-1);
    sigsqold(t) = rho^2*sigsq(t-1) + sig2v;

    %calls x_newtonsolve to find solution to nonlinear posterior prediction estimate
    [xhat(t),flagfail] = x_newtonsolve(muone,  xhatold(t), sigsqold(t), cornum(t-1), ...
         Z(t-1), alpha, beta, rho, gamma, sig2e);

    if flagfail>0
        number_fail = [number_fail t];
    end

    %calculates sigma k squared hat
    sigsq(t) = (1/sigsqold(t) + beta^2/sig2e + gamma^2*exp(muone+gamma*xhat(t))/(1+exp(muone+gamma*xhat(t)))^2)^-1 ;
end

if isempty(number_fail)<1
    fprintf(2,'Newton convergence failed at times %d \n', number_fail)
    failed=1;
    p=0;
    xhat=0;
    sigsq=0;
    xhatold=0;
    sigsqold=0;
    return;
end




