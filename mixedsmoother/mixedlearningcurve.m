function [alph, beta, gamma, rho, sig2e, sig2v, xnew, signewsq, muone, a] = mixedlearningcurve(N, Z, background_prob, rhog, alphag, betag, sig2eg, sig2vg)
%MIXEDLEARNINGCURVE  EM learning-curve estimation for mixed binary/continuous data
%
%   Usage:
%       [alph, beta, gamma, rho, sig2e, sig2v, xnew, signewsq, muone, a] = ...
%           mixedlearningcurve(N, Z, background_prob, rhog, alphag, betag, ...
%                              sig2eg, sig2vg)
%
%   Inputs:
%       N               : 2xK double - row1: number correct per trial; row2: max possible -- required
%       Z               : 1xK double - continuous observation (e.g. reaction time) -- required
%       background_prob : double - chance-level probability of correct response -- required
%       rhog            : double - initial guess for state AR(1) coefficient rho -- required
%       alphag          : double - initial guess for RT intercept alpha -- required
%       betag           : double - initial guess for RT slope beta -- required
%       sig2eg          : double - initial guess for RT noise variance sig2e -- required
%       sig2vg          : double - initial guess for state variance sig2v -- required
%
%   Outputs:
%       alph     : double - EM estimate of RT intercept
%       beta     : double - EM estimate of RT slope
%       gamma    : double - binary observation state weight (fixed to 0 here)
%       rho      : double - EM estimate of state AR(1) coefficient
%       sig2e    : double - EM estimate of RT noise variance
%       sig2v    : double - EM estimate of state variance
%       xnew     : 1xK+1 double - smoothed state estimate x{k|K}
%       signewsq : 1xK+1 double - smoothed state variance SIG^2{k|K}
%       muone    : double - logit of background probability
%       a        : 1xK+1 double - smoother gain A{k}
%
%   Notes:
%       Iterates forward filter (recfilter), backward smoother (backest), and
%       M-step (m_step) for up to 3000 iterations with convergence criterion
%       1e-6 on the mean absolute change in [alpha, beta, sig2e, sig2v].
%       Originally authored by Anne Smith (Oct 15, 2003); updated by
%       Anne Smith (Nov 29, 2010) and Michael Prerau.
%
%   See also: mixedlearningcurve2, recfilter, backest, m_step
%
%   ∿∿∿  Prerau Laboratory MATLAB Codebase · sleepEEG.org  ∿∿∿
%        Source: https://github.com/preraulab/labcode_main

stats = [];
xfilt=[];
cornum = N(1,:);
totnum = N(2,:);


%PARAMETERS
%starting guess for rho
rho = rhog;
%starting guess for beta
beta = betag;
%starting guess for alpha
alph = alphag;
%starting guess for sige = sqrt(sigma_eps squared)
sig2e = sig2eg;
%starting guess for sige = sqrt(sigma_v squared)
sig2v = sig2vg;
gamma=0;
%set the value of mu from the chance of correct
muone = log(background_prob/(1-background_prob)) ;

%convergence criterion for sigma_eps_squared
cvgce_crit = 1e-6;

%----------------------------------------------------------------------------------
%loop through EM algorithm

xguess = 0;  %starting point for random walk x


for jk=1:3000

    %forward filter
    [xfilt, sfilt, xold, sold] = ...
        recfilter(N, Z, sig2e, sig2v, xguess, muone, rho, beta, alph, gamma);

    %backward filter
    [xnew, signewsq, a] = backest(xfilt, xold, sfilt, sold);

    %maximization step
    [alph, beta, gamma, rho, sig2e, sig2v, xnew, muone] = ...
         m_step(N, Z, signewsq, xnew, a, muone);


    newsigsq(jk) = sig2v;

    signewsq(1) = sig2v;    %updates the initial value of the latent process variance

    xnew1save(jk) = xnew(1);

    %check for convergence of parameters
    stats = [stats; [alph beta sig2e sig2v]] ;
    if(jk>1)
        diffsv = stats(jk,:) - stats(jk-1,:);
        a1   = mean(abs(diffsv));
        if( a1 < cvgce_crit )
            fprintf(2, 'EM converged after %d  \n', jk)
            break
        end
    end

    xguess = xnew(1);

end

if(jk == 3000)
    fprintf(2,'failed to converge after %d steps; convergence criterion was %f \n', jk, cvgce_crit)
end
failed=0;
fprintf(2,' alpha is %f, beta is %f, sigesq is %f, sigvsq is %f \n', alph, beta, sig2e, sig2v);

