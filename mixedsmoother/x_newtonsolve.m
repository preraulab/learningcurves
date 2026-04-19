function [xres, timefail] = x_newtonsolve(muone, xold, sig2old, cornum, z, alpha, beta, rho,gamma, sig2e);
%X_NEWTONSOLVE  Newton solver for the nonlinear posterior mode in the mixed model
%
%   Usage:
%       [xres, timefail] = x_newtonsolve(muone, xold, sig2old, cornum, z, ...
%                                         alpha, beta, rho, gamma, sig2e)
%
%   Inputs:
%       muone   : double - logit of background (chance) probability -- required
%       xold    : double - previous-trial state estimate x{k-1|k-1} -- required
%       sig2old : double - one-step prediction variance SIG^2{k|k-1} -- required
%       cornum  : double - binary/count observation at current trial -- required
%       z       : double - continuous (reaction-time) observation at current trial -- required
%       alpha   : double - RT observation intercept -- required
%       beta    : double - RT observation slope -- required
%       rho     : double - state AR(1) coefficient -- required
%       gamma   : double - binary observation state weight -- required
%       sig2e   : double - RT observation noise variance -- required
%
%   Outputs:
%       xres     : double  - converged posterior mode estimate x{k|k}
%       timefail : logical - 1 if Newton iteration failed to converge, 0 otherwise
%
%   Notes:
%       Uses up to 200 Newton iterations with tolerance 1e-14 on |x_{i+1} - x_i|.
%
%   See also: recfilter, newtonsolve, mixedlearningcurve
%
%   ∿∿∿  Prerau Laboratory MATLAB Codebase · sleepEEG.org  ∿∿∿

timefail = 1; %time when the algorithm fails

%Set the initial guess for x hat to the old value of x
x(1)=xold-rho*xold-((sig2old*beta)/(sig2old*beta^2+sig2e))*(z-alpha-beta*rho*xold)-...
    ((sig2e*sig2old)/(sig2old*beta^2+sig2e))*...
    (cornum - exp(muone)*exp(gamma*xold)/(1+exp(muone)*exp(gamma*xold)));

for i = 1:200
    %Find x hat
    g(i) = x(i)-rho*xold-((sig2old*beta)/(sig2old*beta^2+sig2e))*(z-alpha-beta*rho*x(i))-...
        ((sig2e*sig2old)/(sig2old*beta^2+sig2e))*...
        (cornum - exp(muone)*exp(gamma*x(i))/(1+exp(muone)*exp(gamma*x(i))));

    %Find the first derivative
    gprime(i) = 1+((sig2e*sig2old)/(sig2old*beta^2+sig2e))*...
        (gamma*exp(muone+gamma*x(i)))/(1+exp(muone+gamma*x(i)))^2;

    %newton's method
    x(i+1)=x(i)-g(i)/gprime(i);
    xres=x(i+1); %Save the result

    %Check for convergence to zero
    if abs(xres-x(i))<1e-14
        timefail = 0;
        return
    end
end
if(i==200)
    timefail = 1;
    return
end
