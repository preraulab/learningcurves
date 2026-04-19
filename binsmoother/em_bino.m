function [newsigsq] = em_bino(I, xnew, signewsq, A, startflag);
%EM_BINO  EM update of state-process variance for the binary learning model
%
%   Usage:
%       newsigsq = em_bino(I, xnew, signewsq, A, startflag)
%
%   Inputs:
%       I         : 2xK double - row1: number correct per trial; row2: max possible -- required
%       xnew      : 1xK+1 double - smoothed state estimate x{k|K} -- required
%       signewsq  : 1xK+1 double - smoothed state variance SIG^2{k|K} -- required
%       A         : 1xK+1 double - smoother gain A{k} -- required
%       startflag : integer - initial-condition rule (0, 1, 2) -- required
%
%   Outputs:
%       newsigsq : double - EM estimate of learning-state process variance SIG_EPSILON^2
%
%   Notes:
%       Computes sufficient statistics W{k|K}, W{k,k-1|K} (equations A.13-A.15)
%       and applies equation A.16 from Smith et al., J Neurosci 2004. The
%       initial-condition terms (term3, term4) depend on startflag:
%         0 -> fixed initial condition at zero
%         1 -> partial update (1.5 * x(2)^2)
%         2 -> estimated initial condition (x(0) = x(1))
%
%   See also: binsmoother, backwardfilter, m_step
%
%   ∿∿∿  Prerau Laboratory MATLAB Codebase · sleepEEG.org  ∿∿∿

M           = size(xnew,2);

xnewt      = xnew(3:M);
xnewtm1    = xnew(2:M-1);
signewsqt  = signewsq(3:M);
A          = A(2:end);

covcalc    = signewsqt.*A;

term1      = sum(xnewt.^2) + sum(signewsqt);
term2      = sum(covcalc) + sum(xnewt.*xnewtm1);

if startflag == 1
 term3      = 1.5*xnew(2)*xnew(2) + 2.0*signewsq(2);
 term4      = xnew(end)^2 + signewsq(end);
elseif( startflag == 0)
 term3      = 2*xnew(2)*xnew(2) + 2*signewsq(2);
 term4      = xnew(end)^2 + signewsq(end);
elseif( startflag == 2)
 term3      = 1*xnew(2)*xnew(2) + 2*signewsq(2);
 term4      = xnew(end)^2 + signewsq(end);
 M = M-1;
end

newsigsq   = (2*(term1-term2)+term3-term4)/M;
