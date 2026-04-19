function [p05, p95, pmid, pmode, pmatrix, xnew, signewsq]=binsmoother(Responses, SigE, BackgroundProb, NumberSteps)
%BINSMOOTHER  EM learning-curve estimation for binary (correct/incorrect) trial data
%
%   Usage:
%       [p05, p95, pmid, pmode, pmatrix, xnew, signewsq] = ...
%           binsmoother(Responses, SigE, BackgroundProb, NumberSteps)
%
%   Inputs:
%       Responses      : 1xK double - number correct per trial (row or column) -- required
%       SigE           : double - initial guess for state random-walk SD -- required
%       BackgroundProb : double - chance-level probability of correct response -- required
%       NumberSteps    : integer - maximum number of EM iterations -- required
%
%   Outputs:
%       p05      : 1xK+1 double - lower 5 percent confidence bound on p(correct)
%       p95      : 1xK+1 double - upper 95 percent confidence bound on p(correct)
%       pmid     : 1xK+1 double - median p(correct)
%       pmode    : 1xK+1 double - mode of the p(correct) density
%       pmatrix  : K+1x1 double - certainty that performance exceeds chance per trial
%       xnew     : 1xK+1 double - smoothed latent-state estimate x{k|K}
%       signewsq : 1xK+1 double - smoothed state variance SIG^2{k|K}
%
%   Notes:
%       Runs forward filter -> backward smoother -> EM M-step until the change in
%       the estimated process variance (and initial state, for UpdaterFlag >= 1)
%       falls below 1e-8. UpdaterFlag is hard-coded to 2 (no prior chance bias).
%       MaxResponse is inferred from max(Responses).
%
%   See also: forwardfilter, backwardfilter, em_bino, pdistn
%
%   ∿∿∿  Prerau Laboratory MATLAB Codebase · sleepEEG.org  ∿∿∿
%        Source: https://github.com/preraulab/labcode_main

MaxResponse=max(Responses);
UpdaterFlag = 2;  %default allows bias

% check data format.  Reshape dataset if needed
[a,b] = size(Responses);
if a>b
    Responses = Responses';
end

I = [Responses; MaxResponse*ones(1,length(Responses))];

SigsqGuess  = SigE^2;

%set the value of mu from the chance of correct
 mu = log(BackgroundProb/(1-BackgroundProb));

%convergence criterion for SIG_EPSILON^2
 CvgceCrit = 1e-8;

%----------------------------------------------------------------------------------

xguess         = 0;


%loop through EM algorithm: forward filter, backward filter, then
%M-step
for i=1:NumberSteps

   %Compute the forward (filter algorithm) estimates of the learning state
   %and its variance: x{k|k} and sigsq{k|k}
   [p, x, s, xold, sold] = forwardfilter(I, SigE, xguess, SigsqGuess, mu);

   %Compute the backward (smoothing algorithm) estimates of the learning
   %state and its variance: x{k|K} and sigsq{k|K}
   [xnew, signewsq, A]   = backwardfilter(x, xold, s, sold);

   if (UpdaterFlag == 1)
        xnew(1) = 0.5*xnew(2);   %updates the initial value of the latent process
        signewsq(1) = SigE^2;
   elseif(UpdaterFlag == 0)
        xnew(1) = 0;             %fixes initial value (no bias at all)
        signewsq(1) = SigE^2;
   elseif(UpdaterFlag == 2)
        xnew(1) = xnew(2);       %x(0) = x(1) means no prior chance probability
        signewsq(1) = signewsq(2);
   end

   %Compute the EM estimate of the learning state process variance
   [newsigsq(i)]         = em_bino(I, xnew, signewsq, A, UpdaterFlag);

   xnew1save(i) = xnew(1);

   %check for convergence
   if(i>1)
      a1 = abs(newsigsq(i) - newsigsq(i-1));
	  a2 = abs(xnew1save(i) -xnew1save(i-1));
      if( a1 < CvgceCrit & a2 < CvgceCrit & UpdaterFlag >= 1)
          fprintf(2, 'EM estimates of learning state process variance and start point converged after %d steps   \n',  i)
          break
      elseif ( a1 < CvgceCrit & UpdaterFlag == 0)
          fprintf(2, 'EM estimate of learning state process variance converged after %d steps   \n',  i)
          break
      end
   end

   SigE   = sqrt(newsigsq(i));
   xguess = xnew(1);
   SigsqGuess = signewsq(1);

end

if(i == NumberSteps)
     fprintf(2,'failed to converge after %d steps; convergence criterion was %f \n', i, CvgceCrit)
end

%-----------------------------------------------------------------------------------
%integrate and do change of variables to get confidence limits

[p05, p95, pmid, pmode, pmatrix] = pdistn(xnew, signewsq, mu, BackgroundProb);

%-------------------------------------------------------------------------------------
%find the last point where the 90 interval crosses chance
%for the backward filter (cback)

cback = find(p05 < BackgroundProb);

if(~isempty(cback))
  if(cback(end) < size(I,2) )
       cback = cback(end);
  else
       cback = NaN;
  end
else
  cback = NaN;
end
