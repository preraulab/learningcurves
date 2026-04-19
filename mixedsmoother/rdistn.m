function [r05, r95, rmid, rmode, rmatrix] = rdistn(x, s, background_prob, alpha, beta);
%RDISTN  Confidence limits for reaction-time under the log-normal observation model
%
%   Usage:
%       [r05, r95, rmid, rmode, rmatrix] = rdistn(x, s, background_prob, alpha, beta)
%
%   Inputs:
%       x               : 1xK double - smoothed state estimate x{k|K} -- required
%       s               : 1xK double - smoothed state variance SIG^2{k|K} -- required
%       background_prob : double - chance-level probability (unused; kept for API) -- required
%       alpha           : double - RT observation intercept -- required
%       beta            : double - RT observation slope -- required
%
%   Outputs:
%       r05     : 1xK double - lower 5 percent RT bound
%       r95     : 1xK double - upper 95 percent RT bound
%       rmid    : 1xK double - median RT
%       rmode   : 1xK double - mode of the RT density
%       rmatrix : Kx1 double - 1 - CDF column at pr = rmid(1), one entry per trial
%
%   Notes:
%       RT is modeled as log-normal: r = exp(alpha + beta * x + eps). The density
%       over pr = 1e-2:1e-2:50 is integrated via cumtrapz to recover quantiles.
%
%   See also: xdistn, binpdistn, pdistn
%
%   ∿∿∿  Prerau Laboratory MATLAB Codebase · sleepEEG.org  ∿∿∿

rmatrix = [];
rmid = [];
rmode = [];
    dels=1e-2;

    pr  = dels:dels:50;
for ov = 1:size(x,2)

    xx = x(ov);
    ss = s(ov);



    pd=(abs(1./(pr*beta).*(sqrt(1/(2*pi*ss)).*exp((-1/2)*((((log(pr)-alpha)./beta)-xx).^2)./ss))));
    sumpd = dels*cumtrapz(pd);
    tot=sumpd(end);

    lowlimit  = find(sumpd>0.05*tot);
    if(~isempty(lowlimit) )
        lowlimit  = lowlimit(1);
    else
        lowlimit  = 1;
    end

    highlimit = find(sumpd>0.95*tot);

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


    r05(ov)   = pr(lowlimit(1));
    r95(ov)   = pr(highlimit(1));
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

inte = fix(rmid(1)/dels);

rmatrix = 1-rmatrix(:, inte);
