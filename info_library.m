classdef info_library
    methods(Static)

% ﻿MATLAB code for calculating local mutual information,
%  total correlation, dual total correlation, and O-Information.
% This is for use when calculating the local O of a smaller subset of a
% larger dataset (i.e. when doing a single subject in a dataset of 95
% subjects). X is the time series of interest-- it will return a local O
% time series of the same length as X. And X_full is the full length time
% series of all runs and subjects concatenated.

% Calculates the probability density of a point pulled from a univariate Gaussian. 
% Arguments: 
%     x: A floating point value: the instantanious value of the z-scored timeseries.
%     mu: A floating point value: The mean of the distribution P(X).
%     sigma: A floating point value: the standard deviation of the distribution P(X)


function pd = gaussian(x,mu,sigma)
    pd = 1/(sigma * sqrt(2*pi)) * exp(-0.5*((x-mu)/sigma)^2);
end

% The shannon information content of a single variable.
% Essentially just a -log() wrapper for the _gaussian() function.
% Takes all the same arguments.
   
function LE = local_entropy(x,mu,sigma)
    LE = -log(lib2_1.gaussian(x,mu,sigma));
end

% Calculates the probability of a 2-dimensional vector pulled from a bivariate Gaussian. 
% Arguments:
%     x: A 2-by-one array giving the joint state of our 2-dimensional system.
%     mu: A 2-by-one array, giving the mean of each dimension. 
%     sigma: A 2-by-one array giving the standard deviation of each dimension.
%     rho: The Pearson correlation between both dimensions. 

function pd2d = gaussian_2d(x,mu,sigma,rho)
    norm = 1/(2*pi*sqrt(1-(rho^2)));
    exp_upper = (x(1)^2) - (2*rho*x(1)*x(2)) + (x(2)^2);
    exp_lower = 2 * (1-(rho^2));
    pd2d = norm * exp(-1 *(exp_upper/exp_lower));
end

% The shannon information content of a single 2-dimensional variable.
% Essentially just a -log() wrapper for the _gaussian_2d() function. 

function LE2d = local_entropy_2d(x,mu,sigma,rho)
    LE = -log(lib2_1.gaussian_2d(x,mu,sigma,rho));
end

% Calculates the local mutual inforamtion a 2-dimensional vector pulled from a bivariate Gaussian.
% Arguments:
%     x: A 2-cell array giving the joint state of our 2-dimensional system.
%     mu: A 2-cell array, giving the mean of each dimension. 
%     sigma: A 2-cell array giving the standard deviation of each dimension.
%     rho: The Pearson correlation between both dimensions. 

function pmi = gaussian_pmi(vec,mu,sigma,rho)
    marg_x = lib2_1.gaussian(vec(1),mu(1),sigma(1));
    marg_y = lib2_1.gaussian(vec(2),mu(2),sigma(2));
    joint = lib2_1.gaussian_2d(vec,mu,sigma,rho);
    pmi = log(joint / (marg_x * marg_y));
end

% Given a 2-dimensional array (channels x time), returns the edge-timeseries array,
% Which has dimensions (((channels**2)-channels)/2, time)

function ets = local_edge_timeseries(X,varargin)
    if numel(varargin)>0
        X_full = varargin{1};
    else
        X_full = X;
    end
    
    N = floor(((size(X,1)^2) - size(X,1))/2);
    ets = zeros(N,size(X,2));
    counter = 1;
    vec = zeros(2,1);
    mu = zeros(2,1);
    sigma = zeros(2,1);
    
    for i = 1:size(X,1)
        mu(1,:) = mean(X_full(i,:));
        sigma(1,:) = std(X_full(i,:));
        
        for j=1:i-1
               
                mu(2,:) = mean(X_full(j,:));
                sigma(2,:) = std(X_full(j,:));
            
                rho = corr(X_full(i,:)',X_full(j,:)','type','Pearson');
            
                for k= 1:size(X,2)
                    vec(1) = X(i,k);
                    vec(2) = X(j,k);
                
                    ets(counter,k) = lib2_1.gaussian_pmi(vec,mu,sigma,rho);
                end
                counter = counter +1;
 
            
        end
    end
end

% Given an N-dimensional array, calculates the joint Shannon information content of every column 
% in the array. 

function joint_ents = local_gaussian_joint_entropy(X,mu,sigma)

    joint_ps = mvnpdf(X',mu',sigma)';
    joint_ents = -log(joint_ps);
    
end

%Returns the local total correlation (integration) for every column in a muldimensional
%timeseries X. 

function LTC = local_total_correlation(X,varargin)
    if numel(varargin)>0
        X_full = varargin{1};
    else
        X_full = X;
    end
    
    N0 = size(X,1);
    N1 = size(X,2);
    
    mu = mean(X_full,2);
    sigma = std(X_full,0,2);
    
    joint_ents = lib2_1.local_gaussian_joint_entropy(X,mu,cov(X_full'));
    sum_marg_ents = zeros(1,N1);
    
    for i = 1:N1
        for j = 1:N0
            sum_marg_ents(i) = sum_marg_ents(i) + lib2_1.local_entropy(X(j,i),mu(j),sigma(j));
        end
    end
    
    LTC = sum_marg_ents - joint_ents;
end

% Returns the local dual total correlation for a multidimensional timeseries X. 

function local_dtc = local_dual_total_correlation(X,varargin)
    if numel(varargin)>0
        X_full = varargin{1};
    else
        X_full = X;
    end
    
    
    N0 = size(X,1);
    N1 = size(X,2); 
    mu = mean(X_full,2);
    
    joint_ents = lib2_1.local_gaussian_joint_entropy(X,mu,cov(X_full'));
    
    sum_resid_ents = zeros(1,N1);
    local_dtc = zeros(1,N1);
    
    for i = 1:N0
        ff = linspace(1,N0,N0); ff(ff==i)=[];
        X_resid = X(ff,:);
        X_resid_full = X_full(ff,:);
        mu = mean(X_resid_full,2);
        joint_ents_resid = lib2_1.local_gaussian_joint_entropy(X_resid,mu,cov(X_resid_full'));
        
        for j = 1:N1
            sum_resid_ents(j) = sum_resid_ents(j) + joint_ents(j) - joint_ents_resid(j);
        end
    end
    
    for i = 1:N1
        local_dtc(i) = local_dtc(i) + joint_ents(i) - sum_resid_ents(i);
    end
end

% Returns the local O-information for every frame in an N-dimensional timeseries X. 
% Frames can be redundency dominated (O > 0) or synergy dominated (O < 0). 

function LO = local_o_information(X,varargin)
    if numel(varargin)>0
        X_full = varargin{1};
    else
        X_full = X;
    end
    
    LO = lib2_1.local_total_correlation(X,X_full) - lib2_1.local_dual_total_correlation(X,X_full);
end

% Returns the S-information (also called the "very mutual information") by James et al. (2011)

function LS = local_s_information(X,varargin)
    if numel(varargin)>0
        X_full = varargin{1};
    else
        X_full = X;
    end
    
    LS = lib2_1.local_total_correlation(X,X_full) + lib2_1.local_dual_total_correlation(X,X_full);
end


%    Gives I(X;Y) where X and Y are two 1-dimensional, z-scored timeseries. 
%    Closed-form conversion of a Pearson correlation coefficient to MI.  

function g_mi = gaussian_mi(X,Y)
    rho = corr(X',Y','type','Pearson');
    g_mi = -0.5*log(1-(rho^2));
end

end
end
