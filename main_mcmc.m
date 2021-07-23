clear;

phis = [0,30,60,90,300,330];
item = load(['hybrid', num2str(phis(1)) ,'.txt']);

N = size(item,1);
P = length(phis);

Nh = N/2;
% Nd = (1:N)-(N+1)/2;
% xp = grid_norm(1:N,N);
xp = meshgrid(1:N,N);
ds_idx = 1:2:N; % Downsampling idxxd

[xd,yd] = meshgrid(xp,xp);
proj = zeros(N,length(phis));

for i=1:length(phis)
    phi = phis(i);
    item = load(['hybrid', num2str(phi) ,'.txt']);
    proj(:,i) = item(:,2);
end

%%
x = [xd(:), yd(:)];
phis = deg2rad(phis);
% [V,I] = max(mean(proj,1));
proj = proj./mean(proj,1)*max(mean(proj,1));

%% Rotation initialize

mat = ones(N,N)*100;
mat(sqrt(xd.^2+yd.^2)>=Nh-1) = 0;

%  surf(xd(ds_idx,ds_idx), yd(ds_idx,ds_idx), mat(ds_idx,ds_idx))

%% set prior
% mu ~ uniform
% A ~ normal
% sigma ~ Inverse Gamma
% mu_prior = unifpdf(xp,min(xp),max(xp));
% A_prior = normpdf(xp,150,2);
% sigma_e_prior = 1./gampdf(xp,10,2);

%% initialising parameters
% mu = randperm(512,5)-255;
mu = unifrnd(min(xp),max(xp),[1,5]);
A = unifrnd(0,300,[1,5]);
sigma_e = 20;




%% my model 

y = proj(:,1)';

% y_hat = 0;
k=3; %number of peak
% 
% for i=1:k
%     y_hat = y_hat + A(i)*normpdf(xp,mu(i),2);
% end

%% likelihood
% likelihd = normpdf(y-y_hat,0,sigma_e);
% lkh = likelihd(x,y,mu,A,sigma_e);
%% posterior 
%{
mu_sig = 20;

mu_prior = unifpdf(mu(k),min(xp),max(xp));
%         mu_star = unifrnd(min(xp),max(xp));
mu_star = mu;
% mu_star(1) = unifrnd(min(xp),max(xp));
mu_star(1) = normrnd(mu(1),mu_sig);
mu_star_prior = unifpdf(mu_star(k),min(xp),max(xp));

w = likelihd(xp,y,mu_star,A,sigma_e)+log(mu_star_prior)...
            -likelihd(xp,y,mu,A,sigma_e)-log(mu_prior);
a = min(log(1),w);
u = rand;

if log(u) < a
    mu = mu_star;
end
%}

K=5;
rng(100);
mu_sig = 20;
A_sig = 10;
sigma_e_sig = 5;
for i=1:500
    for k=1:3
        
        % mu update
        mu_prior = unifpdf(mu(k),min(xp),max(xp));
        mu_star = mu;
        % mu_star(1) = unifrnd(min(xp),max(xp));
        mu_star(k) = normrnd(mu(k),mu_sig);
        mu_star_prior = unifpdf(mu_star(k),min(xp),max(xp));

        w = likelihd(xp,y,mu_star,A,sigma_e)+log(mu_star_prior)...
            -likelihd(xp,y,mu,A,sigma_e)-log(mu_prior);
        a = min(log(1),w);
        u = rand;

        if log(u) < a
            mu = mu_star;
        end
        
        % update A
        A_star = A;
        A_star(k) = normrnd(A(k),A_sig);
        
        A_prior = normpdf(A_star(k),A(k),A_sig);
        A_star_prior = normpdf(A(k),A_star(k),A_sig);

        w = likelihd(xp,y,mu,A_star,sigma_e)+log(A_star_prior)...
            -likelihd(xp,y,mu,A,sigma_e)-log(A_prior);
        a = min(log(1),w);
        u = rand;

        if log(u) < a
            A = A_star;
        end
        
        % update sigma_e
        sigma_e_star = normrnd(sigma_e,sigma_e_sig);
        
        sigma_e_prior = 1./gampdf(sigma_e,10,2);
        sigma_e_star_prior = 1./gampdf(sigma_e_star,10,2);

        w = likelihd(xp,y,mu,A,sigma_e_star)+log(sigma_e_star_prior)...
            -likelihd(xp,y,mu,A,sigma_e)-log(sigma_e_prior);
        a = min(log(1),w);
        u = rand;

        if log(u) < a
            sigma_e = sigma_e_star;
        end
        K= length(mu);
        y_hat = 0;
        for j=1:K
            y_hat = y_hat + A(j)*normpdf(xp,mu(j),2);
        end
        figure(1); clf
        plot(y); hold on
        plot(y_hat)
        pause(0.1);
        
    end
    
    
end

y_hat = 0;
for i=1:k
    y_hat = y_hat + A(i)*normpdf(xp,mu(i),2);
end


hold on
plot(xp,y)
plot(xp,y_hat)

