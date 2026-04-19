function  [alph, beta, gamma, rho, sig2e, sig2v, xnew, muone] ...
    = m_step(N, Z, signewsq, xnew, a, muone, startflag, binflag)
%M_STEP  EM maximization step for the mixed binary/continuous learning-curve model
%
%   Usage:
%       [alph, beta, gamma, rho, sig2e, sig2v, xnew, muone] = ...
%           m_step(N, Z, signewsq, xnew, a, muone, startflag, binflag)
%
%   Inputs:
%       N        : 1xK double - binary/count observations per trial -- required
%       Z        : 1xK double - continuous observations (e.g. RT) -- required
%       signewsq : 1xK+1 double - smoothed state variance SIG^2{k|K} -- required
%       xnew     : 1xK+1 double - smoothed state estimate x{k|K} -- required
%       a        : 1xK+1 double - smoother gain A{k} -- required
%       muone    : double - logit of background probability (pass-through) -- required
%       startflag: integer - initial-condition rule (0: fixed, 2: estimated) -- required
%       binflag  : logical - if true, suppress RT model (force alpha=beta=0) -- required
%
%   Outputs:
%       alph     : double - updated RT intercept
%       beta     : double - updated RT slope
%       gamma    : double - binary observation state weight (fixed to 1)
%       rho      : double - state AR(1) coefficient (fixed to 1)
%       sig2e    : double - updated RT noise variance
%       sig2v    : double - updated state variance
%       xnew     : 1xK+1 double - smoothed state (passed through)
%       muone    : double - logit of background probability (passed through)
%
%   Notes:
%       Implements the closed-form M-step updates using sufficient statistics
%       W{k|K}, W{k-1|K}, W{k,k-1|K}. Sensitive to the initial-condition
%       specification via startflag. gamma and rho are held fixed.
%
%   See also: em_bino, mixedlearningcurve, mixedlearningcurve2
%
%   ∿∿∿  Prerau Laboratory MATLAB Codebase · sleepEEG.org  ∿∿∿
%        Source: https://github.com/preraulab/labcode_main


K = length(N);

gamma=1; %fixed

%added by ACS 12/02/2010 to deal with different ics on x
%EM convergence is very sensitive to specification of this part
M          = K+1;
xnewt      = xnew(3:M);
xnewtm1    = xnew(2:M-1);
signewsqt  = signewsq(3:M);
A          = a(2:end);
covcalc    = signewsqt.*A;
term1      = sum(xnewt.^2) + sum(signewsqt);
term2      = sum(covcalc) + sum(xnewt.*xnewtm1);

if (startflag  == 0)                   %fixed initial condition
    term3      = 2*xnew(2)*xnew(2) + 2*signewsq(2);
    term4      = xnew(end)^2 + signewsq(end);
elseif( startflag == 2)                %estimated initial condition
    term3      = 1*xnew(2)*xnew(2) + 2*signewsq(2);
    term4      = xnew(end)^2 + signewsq(end);
    M = M-1;
end

sig2v   = (2*(term1-term2)+term3-term4)/M;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

WkK = sum(signewsq(2:end)+xnew(2:end).^2);
Wkm1K = xnew(1)^2+sum(signewsq(2:end-1)+xnew(2:end-1).^2);
Wkm1kK =xnew(1)*xnew(2)+sum(a(1:end-1).*signewsq(3:end)+xnew(2:end-1).*xnew(3:end));
%

ab = inv([K sum(xnew(2:end));sum(xnew(2:end)) WkK])*[sum(Z);sum(xnew(2:end).*Z)];
if binflag
    alph = 0;%ab(1);
    beta  = 0; %ab(2);
else
    alph = ab(1);
    beta  = ab(2);
end
sig2e = (1/(K))*(sum(Z.^2)+K*alph^2+beta^2*WkK-2*alph*sum(Z)-2*beta*sum(xnew(2:end).*Z)+2*alph*beta*sum(xnew(2:end)));

rho = 1; %fixed

