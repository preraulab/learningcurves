function  [xhat, sigsq, xhatold, sigsqold] ... 
    = recfilter(N, Z, sig2e, sig2v, xguess, muone, rho, beta, alpha, gamma)

%implements the forward recursive filtering algorithm
%on the spike train data N
%variables:
%        xhatold                    one-step prediction
%        sigsqold                   one-step prediction variance
%        xhat                       posterior mode
%        sigsq                      posterior variance
%        N                          The point process
%        cornum (1 by num_trials)   vector of number correct at each trial N(1,:)
%        totnum (1 by num_trials)   total number that could be correct at each trial
%                                   N(2,:)
%
%        Z                          The reaction time (continuous)
%
%Parameteres:
%        rho
%        beta
%        alpha
%        muone

failed=0;

T = length(N);
cornum = N;


%set up some initial values
xhat(1) = xguess;    
sigsq(1) = sig2v;   

count = 1;

%number_fail saves the time steps if Newton method fails
number_fail = [];

%loop through all time
for t=2:T+1
    xhatold(t)  = rho*xhat(t-1);
    sigsqold(t) = rho^2*sigsq(t-1) + sig2v;
    
    %calls x_newtonsolve to find solution to nonlinear posterior prediction estimate
    [xhat(t),flagfail] = x_newtonsolve(muone,  xhatold(t), sigsqold(t), cornum(t-1), ...
         Z(t-1), alpha, beta, rho, gamma, sig2e);
    
    if flagfail>0
        number_fail = [number_fail t];
    end
    
    %calculates sigma k squared hat
    sigsq(t) = (1/sigsqold(t) + beta^2/sig2e + gamma^2*exp(muone+gamma*xhat(t))/(1+exp(muone+gamma*xhat(t)))^2)^-1 ;
end

if isempty(number_fail)<1
    fprintf(2,'Newton convergence failed at times %d \n', number_fail)
    failed=1;
    p=0;
    xhat=0;
    sigsq=0;
    xhatold=0;
    sigsqold=0;
    return;
end





