function [alph, beta, gamma, rho, sig2e, sig2v, xnew, signewsq, muone, a] ...
    = mixedlearningcurve2(N, Z, background_prob, rhog, alphag, betag, sig2eg, sig2vg, startflag, binflag)
%MIXEDLEARNINGCURVE2  EM learning-curve estimation with configurable initial condition
%
%   Usage:
%       [alph, beta, gamma, rho, sig2e, sig2v, xnew, signewsq, muone, a] = ...
%           mixedlearningcurve2(N, Z, background_prob, rhog, alphag, betag, ...
%                               sig2eg, sig2vg, startflag, binflag)
%
%   Inputs:
%       N               : 1xK double - observations at each trial -- required
%       Z               : 1xK double - continuous observation (e.g. reaction time) -- required
%       background_prob : double - chance-level probability of correct response -- required
%       rhog            : double - initial guess for state AR(1) coefficient rho -- required
%       alphag          : double - initial guess for RT intercept alpha -- required
%       betag           : double - initial guess for RT slope beta -- required
%       sig2eg          : double - initial guess for RT noise variance sig2e -- required
%       sig2vg          : double - initial guess for state variance sig2v -- required
%       startflag       : integer - initial-condition rule: 0 fixes xnew(1)=0 (no bias),
%                         2 sets xnew(1)=xnew(2) (no prior chance probability), other values
%                         leave the smoother initial condition unchanged -- required
%       binflag         : logical - if true, suppress RT model (force alpha=beta=0) -- required
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
%       Variant of mixedlearningcurve with selectable initial-condition handling
%       (startflag) and a binary-only mode (binflag). Iterates up to 3000 EM
%       steps with convergence criterion 1e-6 on mean absolute change in
%       [alpha, beta, sig2e, sig2v]. Originally authored by Anne Smith
%       (Oct 15, 2003); updated by Anne Smith (Nov 29, 2010) and Michael Prerau.
%
%   See also: mixedlearningcurve, recfilter, backest, m_step
%
%   ∿∿∿  Prerau Laboratory MATLAB Codebase · sleepEEG.org  ∿∿∿

stats = [];
xfilt=[];
cornum = N;



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

   if (startflag == 0)
        xnew(1) = 0;             %fixes initial value (no bias at all)
        signewsq(1) = sig2v^2;
   elseif(startflag == 2)
        xnew(1) = xnew(2);       %x(0) = x(1) means no prior chance probability
        signewsq(1) = signewsq(2);
   end

    %maximization step
    [alph, beta, gamma, rho, sig2e, sig2v, xnew, muone] = ...
         m_step(N, Z, signewsq, xnew, a, muone, startflag, binflag);


    newsigsq(jk) = sig2v;

    signewsq(1) = sig2v;    %updates the initial value of the latent process variance

    xnew1save(jk) = xnew(1);

    %check for convergence of parameters
    stats = [stats; [alph beta sig2e sig2v]] ;
    if(jk>1)
        diffsv = stats(jk,:) - stats(jk-1,:);
        a1   = mean(abs(diffsv));
        if( a1 < cvgce_crit )
          %  fprintf(2, 'EM converged after %d  \n', jk)
            break
        end
    end

    xguess = xnew(1);

end

% if(jk == 3000)
%     fprintf(2,'failed to converge after %d steps; convergence criterion was %f \n', jk, cvgce_crit)
% end
failed=0;
% fprintf(2,' alpha is %f, beta is %f, sigesq is %f, sigvsq is %f \n', alph, beta, sig2e, sig2v);

