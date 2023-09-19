%LEARNINGCURVE_PFILTER_PTILE Computes binary, continuous, or mixed
%learning curve using a particle filter DOESN'T SAVE PARTICLES ONLY
%PERCENTILES. This can be used for cases in which the number of particles
%must be very large.
%
%   State model: x_t=x_t-1+Ex, where Ex ~ N(0, sig2x)
%   Binary obs. model: p_t=exp(x_t)/(1+exp(x_t))
%
%   Usage:
%   [param_ests, particles]=binary_learningcurve_pfilter_ptile() -- RUNS EXAMPLE
%   [param_ests, particles]=binary_learningcurve_pfilter_ptile(times, data)
%   [param_ests, particles]=binary_learningcurve_pfilter_ptile(times, data, num_particles, smoother, prog_bar, plot_on)
%
%   Input:
%   times: 1xT vector of corresponding times (in seconds)
%   data: A 1xT matrix of binary (1st row) and continuous (2nd row) observations. Set either row to NaN for just the binary or continuous filter, or use NaN for missing data.
%   num_particles: Number of particles to use (Default: 5000)
%   smoother: 1 = Use forward/backward filtering, 0 = Filter only (Default: 1)
%   progbar: 1 = Display progress bar, 0 = No progress bar (Default: 1);
%   plot on: 1 = Plot output graph, 0 = No output plot (Default: 1);
%
%   Example:
%
%     %---------------------------------------------
%     %RUN WITH NO INPUTS TO EXECUTE THIS EXAMPLE
%     %---------------------------------------------
%
%     %Set up times
%     times=1:300;
%
%     %Start with a high probability of success
%     x=zeros(size(times));
%     x(1)=log(9999);
%
%     %Create a random walk
%     for i=2:length(times)
%         x(i)=x(i-1)-.1+randn*.05;
%     end
%
%     %Compute binary responses
%     pk=exp(x)./(1+exp(x));
%     data=rand(1,length(x))<=pk;
%
%
%     %Compute smoother
%     binary_learningcurve_pfilter_ptile(times, data);
%
%   Copyright 2013 Michael J. Prerau, Ph.D.
%
%   Last modified 12/02/2015
%********************************************************************

function [param_ests, particles]=binary_learningcurve_pfilter_ptile(times, data, num_particles, smoother, prog_bar, plot_on)
%Generate simulated data if no inputs
if nargin==0
    %Set up times
    Fs=1;
    dt=1/Fs;
    num_obs=300;
    times=dt:dt:num_obs;
    
    %Start with a high probability of success
    x=zeros(size(num_obs));
    x(1)=log(999);
    
    %Create a random walk
    for i=2:num_obs
        x(i)=x(i-1)-.1+randn*.05;
    end
    
    %Compute binary responses
    pk=exp(x)./(1+exp(x));
    N=rand(1,length(x))<=pk;
    
    %Create simulated data
    data=nan(1,length(times));
    data(1:Fs:end)=N;
end

%Set up default inputs
if nargin<3
    num_particles=5000;
end

if nargin<4
    smoother=1;
end

if nargin<5
    prog_bar=0;
end

if nargin<6
    plot_on=true;
end

%Handle forward/backward filter
if smoother
    data=[ fliplr(data(2:end))  data];
    times=[ fliplr(times(2:end)) times];
end

%Number of time points
N=length(data);

%Number of parameters to estimate
num_params=2;

%Create <time>x<variable>x<particle> matrix
particles=zeros(num_params,num_particles);
new_particles=particles;

%-----------------------------------------------
%Set Priors and state variances
%-----------------------------------------------
%Set broad prior for each parameter
%X0 (Can be anywhere from 0 to .999 initial prob)
particles(1,:)=rand(1,num_particles)*log(99);
%Sig2x: State variance
particles(2,:)=rand(1,num_particles)*.005;

%Start the progress bar
if prog_bar
    progressbar;
end

%Iterate through all time
for t=2:N+1
    %-----------------------------------------------
    %Update using the one step prediction equation
    %-----------------------------------------------
    %*********PUT ONE-STEP PREDCTIONS HERE*********
    %Sig2x: State variance
    new_particles(2,:)=abs(particles(2,:)+randn(1,num_particles)*.005);
    
    %Xt=Xt-1 + Ex
    new_particles(1,:)=particles(1,:)+randn(1,num_particles).*new_particles(2,:);
    
    
    %----------------------------------------------------------
    %Use the observation model to compute the estimated state
    %----------------------------------------------------------
    x_hat=squeeze(new_particles(1,:));
    
    %Estimated binomial probability
    p_hat=exp(x_hat)./(1+exp(x_hat));
    
    
    %-----------------------------------------------
    %Compute the likelihood/weights
    %-----------------------------------------------
   
    %Get binary observation
    if ~isnan(data(t-1))
        bin_observation=repmat(data(t-1),num_particles,1)';

        %For missing continuous just use binary data
        loglikelihood=bin_observation.*log(p_hat)+(1-bin_observation).*log(1-p_hat);
        
        %Compute the weights
        pweights=loglikelihood;
        
        %-----------------------------------------------
        %Resample Particles
        %-----------------------------------------------
        %Get distribution of weights
        weights=exp(pweights-max(pweights));
        weights(isnan(weights))=0;
        
        %Sample particles given the distribution of the weights
        ind = weighted_random_index(particles(1,:), num_particles, weights);
        new_particles(:,:)=new_particles(:,ind);
    end
    
    particles=new_particles;
    
    x_hat=(new_particles(1,:))';
    
    p_hat_out(t-1,:)=prctile(exp(x_hat)./(1+exp(x_hat)),[2.5, 50 97.5]);
    
    %Show progress if wanted
    if prog_bar & ~mod(round((t-1)/N*100),5)
        progressbar((t-1)/N);
    end
end

%Take only the first half of data if there's a forward/back estimate
if smoother   
    num_obs=(N-1)/2;
    data=data(end-(num_obs-1):end);
    times=times(end-(num_obs-1):end);
    
    p_hat_out=p_hat_out(end-(num_obs-1):end,:);
end

%Get all the parameter estimates
param_ests=p_hat_out;

%Plot figure if option selected
if plot_on
    %Plot the figures
    figure('color','w','units','normalized','position',[0 0 1 1]);
    
    ax(1)=axes('position',[0.0800    0.8900    0.8700    0.0600]);
    ax(2)=axes('position',[0.0800    0.0500    0.8700    0.7600]);
    
    linkaxes(ax,'x');
    
    subplot(ax(1));
    %Get correct and incorrect indices
    binds=~isnan(data(1,:));
    bdata=data(binds);
    btimes=times(binds);
    
    binplot(btimes,bdata);
    set(gca,'ytick',[-.5 .5],'yticklabel',{'Incorr.','Correct'});
    xlabel('Time');
    title('Behavioral Reponses');
    
    subplot(ax(2));
    hold on;
    fill([times fliplr(times)],[p_hat_out(:,1)' fliplr(p_hat_out(:,3)')],[1 .7 .7],'edgecolor','none');
    plot(times, p_hat_out(:,2),'k','linewidth',2);
    axis tight;
    ylabel('P(Response)');
    xlabel('Time');
    title('Behavioral Reponse Probability');
end


