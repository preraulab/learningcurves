function [r025, r975, rmid, rmode, rmatrix] = xdistn(x, s, background_prob, alpha, beta);
%XDISTN  Confidence limits and mode of the learning-state distribution (mixed model)
%
%   Usage:
%       [r025, r975, rmid, rmode, rmatrix] = xdistn(x, s, background_prob, alpha, beta)
%
%   Inputs:
%       x               : 1xK double - smoothed state estimate x{k|K} -- required
%       s               : 1xK double - smoothed state variance SIG^2{k|K} -- required
%       background_prob : double - chance-level probability (unused here; kept for API) -- required
%       alpha           : double - RT observation intercept (unused here; kept for API) -- required
%       beta            : double - RT observation slope (unused here; kept for API) -- required
%
%   Outputs:
%       r025    : 1xK double - lower 2.5 percent bound of the state distribution
%       r975    : 1xK double - upper 97.5 percent bound of the state distribution
%       rmid    : 1xK double - median of the state distribution
%       rmode   : 1xK double - mode of the state distribution
%       rmatrix : Kx1 double - CDF column at p = 0.5, one entry per trial
%
%   Notes:
%       Builds a normal pdf over pr = 1e-3:1e-3:10 per trial and integrates
%       via cumtrapz to derive quantiles.
%
%   See also: rdistn, binpdistn, pdistn
%
%   ∿∿∿  Prerau Laboratory MATLAB Codebase · sleepEEG.org  ∿∿∿


rmatrix = [];
% x=x+abs(min(x))+3;
for ov = 1:size(x,2)

    xx = x(ov);
    ss = s(ov);

    dels=1e-3;

    pr  = dels:dels:10;

    pd=normpdf(pr,x(ov),s(ov));
    sumpd = dels*abs(cumtrapz(pd));
    tot=sumpd(end);

    lowlimit  = find(sumpd>0.025*tot);
    if(~isempty(lowlimit) )
        lowlimit  = lowlimit(1);
    else
        lowlimit  = 1;
    end

    highlimit = find(sumpd>0.975*tot);

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


    r025(ov)   = pr(lowlimit(1));
    r975(ov)   = pr(highlimit(1));
    rmid(ov)  = pr(middlimit(1));
    [y,i]     = max(pd);
    rmode(ov) = pr(i);

    rmatrix =[rmatrix; sumpd];

    %     if rmode(ov)< r05(ov);
    %         r05(ov);
    %         figure;
    %         plot(pd);
    %         pause;
    %     end;
end

inte = abs(fix(.5/dels));

rmatrix = rmatrix(:, inte);
