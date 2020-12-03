function [indices, values] = weighted_random_index(data, num_samples, weights)

%WEIGHTED_RANDOM_INDEX Returns indices of data resampled based on weights
%
%   Usage:
%   [indices, values] = weighted_random_index(data, num_samples, weights)
%
%   Input:
%   data: 1xN vector of values
%   num_samples: Number of samples to estimate
%   weights: 1xN vector of weights
%
%   Example:
%
%   Copyright 2015 Michael J. Prerau, Ph.D. (based on randsamp)
%
%   Last modified 10/01/2015
%********************************************************************
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
