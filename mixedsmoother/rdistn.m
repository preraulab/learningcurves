function [r05, r95, rmid, rmode, rmatrix] = rdistn(x, s, background_prob, alpha, beta);

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
