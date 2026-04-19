function [p05, p95, pmid, pmode, pmatrix] = binpdistn(q, s, muone, background_prob);
%BINPDISTN  Confidence limits for p(correct) under the binary observation model
%
%   Usage:
%       [p05, p95, pmid, pmode, pmatrix] = binpdistn(q, s, muone, background_prob)
%
%   Inputs:
%       q               : 1xK double - smoothed state estimate x{k|K} -- required
%       s               : 1xK double - smoothed state variance SIG^2{k|K} -- required
%       muone           : double - logit of background (chance) probability -- required
%       background_prob : double - chance-level probability used to index pmatrix -- required
%
%   Outputs:
%       p05     : 1xK double - lower 5 percent confidence bound of p(correct)
%       p95     : 1xK double - upper 95 percent confidence bound of p(correct)
%       pmid    : 1xK double - median p(correct)
%       pmode   : 1xK double - mode of the p(correct) density
%       pmatrix : Kx1 double - CDF column at p = background_prob, one entry per trial
%
%   Notes:
%       Probability density for p is obtained by a change-of-variables from the
%       Gaussian state density through the logit link with offset muone.
%
%   See also: pdistn, xdistn, rdistn
%
%   ∿∿∿  Prerau Laboratory MATLAB Codebase · sleepEEG.org  ∿∿∿
%        Source: https://github.com/preraulab/labcode_main


pmatrix = [];
 dels=1e-4;

 pr  = dels:dels:1-dels;
for ov = 1:size(q,2)

 qq = q(ov);
 ss = s(ov);



 fac   = log( pr./(1-pr)/exp(muone) ) - qq;
 fac   = exp(-fac.^2/2/ss);
 pd    = dels*(sqrt(1/2/pi/ss) * 1./(pr.*(1-pr)).* fac);

% sumpd = cumsum(pd);
 sumpd = cumtrapz(pd);

lowlimit  = find(sumpd>0.05);
if(~isempty(lowlimit) )
lowlimit  = lowlimit(1);
else
lowlimit  = 1;
end

highlimit = find(sumpd>0.95);
% highlimit = find(sumpd>0.995);
if(~isempty(highlimit) )
if(length(highlimit)>1)
highlimit = highlimit(1)-1;
else
highlimit =  highlimit(1);
end
else
highlimit = length(pr);
end

middlimit = find(sumpd>0.5);
if(~isempty(middlimit))
middlimit = middlimit(1);
else
middlimit = length(pr);
end


 p05(ov)   = pr(lowlimit(1));
 p95(ov)   = pr(highlimit(1));
 pmid(ov)  = pr(middlimit(1));
 [y,i]     = max(pd);
 pmode(ov) = pr(i);

 pmatrix =[pmatrix; sumpd];


end

inte = fix(background_prob/dels);

pmatrix = pmatrix(:, inte);
