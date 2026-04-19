%WEIGHTED_RANDOM_INDEX  Resample indices of a data vector with replacement using given weights
%
%   Usage:
%       [indices, values] = weighted_random_index(data, num_samples)
%       [indices, values] = weighted_random_index(data, num_samples, weights)
%
%   Inputs:
%       data        : 1xN double - values to sample from -- required
%       num_samples : integer - number of samples to draw -- required
%       weights     : 1xN double - non-negative weights (default: uniform ones)
%
%   Outputs:
%       indices : num_samples x 1 integer - sampled indices into data
%       values  : num_samples x 1 double - data(indices)
%
%   Notes:
%       Weights are normalized internally. Based on the RANDSAMPLE
%       implementation but returns indices explicitly. Uses histc() on
%       cumulative edges for the weighted draw.
%
%   See also: randsampleind, histc
%
%   ∿∿∿  Prerau Laboratory MATLAB Codebase · sleepEEG.org  ∿∿∿

function [indices, values] = weighted_random_index(data, num_samples, weights)

if nargin <3
    weights=ones(1,length(data));
end

%Normalize weights
norm_weights = weights(:)'/sum(weights);

edges = min([0 cumsum(norm_weights)],1); % protect against accumulated round-off
edges(end) = 1; % get the upper edge exact

%Get indices
[~, indices] = histc(rand(num_samples,1),edges);

if nargout==2
    values=data(indices);
end
