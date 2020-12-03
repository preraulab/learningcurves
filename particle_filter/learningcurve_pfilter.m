%LEARNINGCURVE_PFILTER Computes binary, continuous, or mixed learning curve using a particle filter
%
%   State model: x_t=x_t-1+Ex, where Ex ~ N(0, sig2x)
%   Binary obs. model: p_t=exp(x_t)/(1+exp(x_t))
%   Cont obs. model: z_t=alpha + beta * x_t + Ez, where Ez ~ N(0, sig2z)0
%
%   Usage:
%   [param_ests, particles]=learningcurve_pfilter() -- RUNS EXAMPLE
%   [param_ests, particles]=learningcurve_pfilter(times, data)
%   [param_ests, particles]=learningcurve_pfilter(times, data, num_particles, smoother, prog_bar, plot_on)
%
%   Input:
%   times: 1xT vector of corresponding times (in seconds)
%   data: A 2xT matrix of binary (1st row) and continuous (2nd row) observations. Set either row to NaN for just the binary or continuous filter, or use NaN for missing data.
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
%     N=rand(1,length(x))<=pk;
%
%     %Compute continous responses
%     a=1;
%     b=.3;
%     Z=a+b.*x+randn(1,length(x))*.5;
%
%     %Create simulated data
%     data=[N;Z];
%
%     %Compute smoother
%     learningcurve_pfilter(times, data);
%
%   Copyright 2013 Michael J. Prerau, Ph.D.
%
%   Last modified 1/07/2013
%********************************************************************

function [param_ests, particles]=learningcurve_pfilter(times, data, num_particles, smoother, prog_bar, plot_on)
%Generate simulated data if no inputs
if nargin==0
    %Set up times
    times=1:300;
    
    %Start with a high probability of success
    x=zeros(size(times));
    x(1)=log(9999);
    
    %Create a random walk
    for i=2:length(times)
        x(i)=x(i-1)-.1+randn*.05;
    end
    
    %Compute binary responses
    pk=exp(x)./(1+exp(x));
    N=rand(1,length(x))<=pk;
    
    %Compute continous responses
    a=1;
    b=.3;
    Z=a+b.*x+randn(1,length(x))*.5;
    
    %Create simulated data
    data=[N;Z];
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
    data=[data fliplr(data(:,2:end)) ];
    times=[times fliplr(times(2:end))];
end

%Number of time points
N=length(data(1,:));

%Number of parameters to estimate
num_params=5;

%Create <time>x<variable>x<particle> matrix
particles=zeros(N+1,num_params,num_particles);

%-----------------------------------------------
%Set Priors and state variances
%-----------------------------------------------
%Set broad prior for each parameter
%X0 (Can be anywhere from 0 to .999 initial prob)
particles(1,1,:)=rand(1,num_particles)*log(999);
%Alpha
particles(1,2,:)=rand(1,num_particles)*10;
%Beta
particles(1,3,:)=rand(1,num_particles)*10;
%Sig2e: Observation variance
particles(1,4,:)=rand(1,num_particles)*5;
%Sig2x: State variance
particles(1,5,:)=rand(1,num_particles)*.005;

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
    particles(t,5,:)=abs(particles(t-1,5,:)+randn(1,1,num_particles)*.005);
    
    %Xt=Xt-1 + Ex
    particles(t,1,:)=particles(t-1,1,:)+randn(1,1,num_particles).*particles(t,5,:);
    %Alpha
    particles(t,2,:)=particles(t-1,2,:)+randn(1,1,num_particles).*.005;
    %Beta
    particles(t,3,:)=particles(t-1,3,:)+randn(1,1,num_particles).*.005;
    %Sig2e: Observation variance
    particles(t,4,:)=abs(particles(t-1,4,:)+randn(1,1,num_particles)*.005);
    
    %----------------------------------------------------------
    %Use the observation model to compute the estimated state
    %----------------------------------------------------------
    x_hat=squeeze(particles(t,1,:));
    alpha=squeeze(particles(t,2,:));
    beta=squeeze(particles(t,3,:));
    sig2e=squeeze(particles(t,4,:));
    
    %Estimated binomial probability
    p_hat=exp(x_hat)./(1+exp(x_hat));
    %Estimated continuous value
    z_hat=alpha+beta.*x_hat;
    
    %-----------------------------------------------
    %Compute the likelihood/weights
    %-----------------------------------------------
    bin=false;
    cont=false;
    
    %Get binary observation
    if ~isnan(data(1,t-1))
        bin_observation=repmat(data(1,t-1),num_particles,1);
        bin=true;
    end
    
    %Get continous observation
    if size(data,1)>1 && ~isnan(data(2,t-1))
        cont_observation=repmat(data(2,t-1),num_particles,1);
        cont=true;
    end
    
    %If there are both observations, use both
    if bin && cont
        %Joint log-likelihood
        loglikelihood=(bin_observation.*log(p_hat)+(1-bin_observation).*log(1-p_hat))-((cont_observation-z_hat).^2./(2*sig2e));
    elseif bin && ~cont
        %For missing continuous just use binary data
        loglikelihood=bin_observation.*log(p_hat)+(1-bin_observation).*log(1-p_hat);
    elseif cont && ~bin
        loglikelihood=-((cont_observation-z_hat).^2./(2*sig2e));
    end
    
    %Resample if there is actual data
    if bin || cont
        %Compute the weights
        pweights=sum(loglikelihood,2);
        
        %-----------------------------------------------
        %Resample Particles
        %-----------------------------------------------
        %Get distribution of weights
        weights=exp(pweights-max(pweights));
        weights(isnan(weights))=0;
        
        %Sample particles given the distribution of the weights
        %         [~,ind]=randsampleind(squeeze(particles(t,1,:)),num_particles,1,weights);
        ind = weighted_random_index(squeeze(particles(t,1,:)), num_particles, weights);
        particles(t,:,:)=squeeze(particles(t,:,ind));
    end
    
    %Show progress if wanted
    if prog_bar & ~mod(round((t-1)/N*100),5)
        progressbar((t-1)/N);
    end
end

%----------------------------
%Compute the parameter estimates
%----------------------------
%Take only the first half of data if there's a forward/back estimate
if smoother
    N=(N+1)/2;
    
    data=data(:, 1:N);
    times=times(1:N);
    particles=particles(end:-1:N,:,:);
end

x_hat=squeeze(particles(2:end,1,:))';
a_hat=squeeze(particles(2:end,2,:))';
b_hat=squeeze(particles(2:end,3,:))';

p_hat=prctile(exp(x_hat)./(1+exp(x_hat)),[2.5, 50 97.5]);
z_hat=prctile(exp(a_hat+b_hat.*x_hat),[2.5, 50 97.5]);

%Get all the parameter estimates
param_ests=cell(1,num_params+2);
param_ests{1}=p_hat;
param_ests{2}=z_hat;
for i=1:num_params
    param_ests{i+2}=prctile(squeeze(particles(2:end,i,:))',[2.5, 50 97.5]);
end

%Plot figure if option selected
if plot_on
    %Plot the figures
    figure('color','w','units','normalized','position',[0 0 1 1]);
    
    if size(data,1)==1
        ax(1)=axes('position',[0.0800    0.8900    0.8700    0.0600]);
        ax(2)=axes('position',[0.0800    0.0500    0.8700    0.7600]);
    elseif size(data,1)==2
        ax(1)=axes('position',[0.0800    0.8900    0.8700    0.0600]);
        ax(2)=axes('position',[0.0800    0.4700    0.8700    0.3400]);
        ax(3)=axes('position',[0.0800    0.0500    0.8700    0.3400]);
    end
    
    linkaxes(ax,'x');
    
    subplot(ax(1));
    %Get correct and incorrect indices
    binds=~isnan(data(1,:));
    bdata=data(1,binds);
    btimes=times(binds);
    
    binplot(btimes,bdata);
    set(gca,'ytick',[-.5 .5],'yticklabel',{'Incorr.','Correct'});
    xlabel('Time');
    title('Behavioral Reponses');
    
    subplot(ax(2));
    hold on;
    fill([times fliplr(times)],[p_hat(1,:) fliplr(p_hat(3,:))],[1 .7 .7],'edgecolor','none');
    plot(times, p_hat(2,:),'k','linewidth',2);
    axis tight;
    ylabel('P(Response)');
    xlabel('Time');
    title('Behavioral Reponse Probability');
    
    %Plot only if mixed
    if size(data,1)==2
        axes(ax(3))
        hold on;
        fill([times fliplr(times)],[z_hat(1,:) fliplr(z_hat(3,:))],[.7 .7 1],'edgecolor','none');
        plot(times, exp(data(2,:)),'.','markersize',10)
        plot(times, z_hat(2,:),'k','linewidth',2);
        axis tight
        
        ylabel('Reaction Time');
        xlabel('Time');
        title('Reactoion Time');
        
    end
end


