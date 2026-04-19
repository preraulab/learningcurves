function [x, timefail] = newtonsolve(mu,  xold, sigoldsq, N, Nmax);
%NEWTONSOLVE  Newton solver for the nonlinear posterior mode in the binary model
%
%   Usage:
%       [x, timefail] = newtonsolve(mu, xold, sigoldsq, N, Nmax)
%
%   Inputs:
%       mu       : double - logit of background (chance) probability -- required
%       xold     : double - one-step prediction x{k|k-1} -- required
%       sigoldsq : double - one-step prediction variance SIG^2{k|k-1} -- required
%       N        : double - number correct at current trial -- required
%       Nmax     : double - max possible correct at current trial -- required
%
%   Outputs:
%       x        : double  - converged posterior mode estimate x{k|k} (equation A.8)
%       timefail : logical - 1 if Newton iteration failed to converge, 0 otherwise
%
%   Notes:
%       Uses up to 40 Newton iterations with tolerance 1e-14 on |x_{i+1} - x_i|.
%       Equation references follow Smith et al., J Neurosci 2004.
%
%   See also: forwardfilter, x_newtonsolve
%
%   ∿∿∿  Prerau Laboratory MATLAB Codebase · sleepEEG.org  ∿∿∿
%        Source: https://github.com/preraulab/labcode_main

it(1) = xold + sigoldsq*(N - Nmax*exp(mu)*exp(xold)/(1 ...
                                  + exp(mu)*exp(xold)));

for i = 1:40
   g(i)     = xold + sigoldsq*(N - Nmax*exp(mu)*exp(it(i))/...
                              (1+exp(mu)*exp(it(i)))) - it(i);
   gprime(i)= -Nmax*sigoldsq*exp(mu)*exp(it(i))/(1+exp(mu)*exp(it(i)))^2 - 1;
   it(i+1)  = it(i) - g(i)/gprime(i);

   x        = it(i+1);
   if abs(x-it(i))<1e-14
      timefail = 0;
      return
   end
end

if(i==40)
   fprintf(2, 'failed to converge \n');
   timefail = 1;
   return
end
