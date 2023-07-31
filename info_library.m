classdef info_library
    methods(Static)

% ﻿MATLAB code for calculating local mutual information,
%  total correlation, dual total correlation, and O-Information.

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
    LE = -log(lib.gaussian(x,mu,sigma));
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
    LE2d = -log(lib.gaussian_2d(x,mu,sigma,rho))
end

% Calculates the local mutual inforamtion a 2-dimensional vector pulled from a bivariate Gaussian.
% Arguments:
%     x: A 2-cell array giving the joint state of our 2-dimensional system.
%     mu: A 2-cell array, giving the mean of each dimension. 
%     sigma: A 2-cell array giving the standard deviation of each dimension.
%     rho: The Pearson correlation between both dimensions. 

function pmi = gaussian_pmi(vec,mu,sigma,rho)
    marg_x = lib.gaussian(vec(1),mu(1),sigma(1));
    marg_y = lib.gaussian(vec(2),mu(2),sigma(2));
    joint = lib.gaussian_2d(vec,mu,sigma,rho);
    pmi = log(joint / (marg_x * marg_y));
end

% Given a 2-dimensional array (channels x time), returns the edge-timeseries array,
% Which has dimensions (((channels**2)-channels)/2, time)

function ets = local_edge_timeseries(X)
    N = floor(((size(X,1)^2) - size(X,1))/2);
    ets = zeros(N,size(X,2));
    counter = 1;
    vec = zeros(2,1);
    mu = zeros(2,1);
    sigma = zeros(2,1);
    
    for i = 1:size(X,1)
        mu(1,:) = mean(X(i,:));
        sigma(1,:) = std(X(i,:));
        
        for j=1:i-1
               
                mu(2,:) = mean(X(j,:));
                sigma(2,:) = std(X(j,:));
            
                rho = corr(X(i,:)',X(j,:)','type','Pearson');
            
                for k= 1:size(X,2)
                    vec(1) = X(i,k);
                    vec(2) = X(j,k);
                
                    ets(counter,k) = lib.gaussian_pmi(vec,mu,sigma,rho);
                end
                counter = counter +1;
 
            
        end
    end
end

function joint_ents = local_gaussian_joint_entropy(X)
    N0 = size(X,1);
    N1 = size(X,2);
    
    mu = mean(X,2);
    sigma = std(X,0,2);
    
    cv = reshape(cov(X',1),[size(X,1),size(X,1)]); 
    iv = inv(cv);
    
    dt = det(cv);
    
    norm = 1/sqrt(((2*pi)^(N0))*dt);
    
    err = zeros(1,N0);
    joint_ents = zeros(1,N1);
    
    for i = 1:N1
        for j = 1:N0
            err(j)= X(j,i)-mu(j);
        end
        
        mul = -0.5 * (err*iv)*err';
        joint_ents(i) = -1*log(norm.* exp(mul));
    end
end

function LTC = local_total_correlation(X)
    N0 = size(X,1);
    N1 = size(X,2);
    
    mu = mean(X,2);
    sigma = std(X,0,2);
    
    joint_ents = lib.local_gaussian_joint_entropy(X);
    sum_marg_ents = zeros(1,N1);
    
    for i = 1:N1
        for j = 1:N0
            sum_marg_ents(i) = sum_marg_ents(i) + lib.local_entropy(X(j,i),mu(j),sigma(j));
        end
    end
    
    LTC = sum_marg_ents - joint_ents;
end

function local_dtc = local_dual_total_correlation(X)
    N0 = size(X,1);
    N1 = size(X,2); 
    
    joint_ents = lib.local_gaussian_joint_entropy(X);
    
    sum_resid_ents = zeros(1,N1);
    local_dtc = zeros(1,N1);
    
    for i = 1:N0
        ff = linspace(1,N0,N0); ff(ff==i)=[];
        X_resid = X(ff,:);
        joint_ents_resid = lib.local_gaussian_joint_entropy(X_resid);
        
        for j = 1:N1
            sum_resid_ents(j) = sum_resid_ents(j) + joint_ents(j) - joint_ents_resid(j);
        end
    end
    
    for i = 1:N1
        local_dtc(i) = local_dtc(i) + joint_ents(i) - sum_resid_ents(i);
    end
end

function LO = local_o_information(X)
    LO = lib.local_total_correlation(X) - lib.local_dual_total_correlation(X);
end

function LS = local_s_information(X)
    LS = lib.local_total_correlation(X) + lib.local_dual_total_correlation(X);
end

function gtc = gaussian_total_correlation(X)
    cor = corr(X','type','Pearson');    
    dt = det(cor);
    gtc = -log(dt)/2;
end

function g_mi = gaussian_mi(X,Y)
    rho = corr(X',Y','type','Pearson');
    g_mi = -0.5*log(1-(rho^2));
end

    end
end


    
    
    


           




