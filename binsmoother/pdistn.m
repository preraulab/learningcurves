function [p05, p95, pmid, pmode, pmatrix] = pdistn(x, s, mu, background_prob);
%PDISTN  Confidence limits for p(correct) under the binary observation model
%
%   Usage:
%       [p05, p95, pmid, pmode, pmatrix] = pdistn(x, s, mu, background_prob)
%
%   Inputs:
%       x               : 1xK double - smoothed state estimate x{k|K} -- required
%       s               : 1xK double - smoothed state variance SIG^2{k|K} -- required
%       mu              : double - logit of background (chance) probability -- required
%       background_prob : double - chance-level probability used to index pmatrix -- required
%
%   Outputs:
%       p05     : 1xK double - lower 5 percent confidence bound on p(correct)
%       p95     : 1xK double - upper 95 percent confidence bound on p(correct)
%       pmid    : 1xK double - median p(correct)
%       pmode   : 1xK double - mode of the p(correct) density
%       pmatrix : Kx1 double - CDF column at p = background_prob (certainty that
%                 performance exceeds chance), one entry per trial
%
%   Notes:
%       The probability density fp{k|j} (equation B.3 in Smith et al. 2004) is
%       obtained by a change of variables from the Gaussian state density
%       through the logit link with offset mu; integration uses cumtrapz on a
%       grid of step size 1e-4.
%
%   See also: binpdistn, xdistn, rdistn
%
%   ∿∿∿  Prerau Laboratory MATLAB Codebase · sleepEEG.org  ∿∿∿

pmatrix = [];

 dels=1e-4;
 pr  = dels:dels:1-dels;
% pr=linspace(0,1,100);

for ov = 1:size(x,2)

 xx = x(ov);
 ss = s(ov);

 term1 = 1./(sqrt(2*pi*ss) * (pr.*(1-pr)));
 term2 = exp(-1/(2*ss) * (log (pr./((1-pr)*exp(mu))) - xx).^2);
 pdf = term1 .* term2;
 pdf = dels * pdf;


% Integrate the pdf
 sumpdf = cumtrapz(pdf);
% sumpdf = cumsum(pdf);

lowlimit  = find(sumpdf>0.05);
if(~isempty(lowlimit) )
lowlimit  = lowlimit(1);
else
lowlimit  = 1;
end

highlimit = find(sumpdf>0.95);
% highlimit = find(sumpdf>0.995);
if(~isempty(highlimit) )
if(length(highlimit)>1)
highlimit = highlimit(1)-1;
else
highlimit =  highlimit(1);
end
else
highlimit = length(pr);
end

middlimit = find(sumpdf>0.5);
if(~isempty(middlimit))
middlimit = middlimit(1);
else
middlimit = length(pr);
end


 p05(ov)   = pr(lowlimit(1));
 p95(ov)   = pr(highlimit(1));
 pmid(ov)  = pr(middlimit(1));
 [y,i]     = max(pdf);
 pmode(ov) = pr(i);


 pmatrix =[pmatrix; sumpdf];

end

inte = fix(background_prob/dels);

pmatrix = pmatrix(:, inte);
