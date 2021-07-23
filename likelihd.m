function lkh = likelihd(xp,y,mu,A,sigma_e)
%LIKELIHD Summary of this function goes here
%   Detailed explanation goes here
% mu = symsum(A*normpdf(),i,[1 k])

K = length(mu);
N = length(xp);
y_hat = 0; % vector
lkh = 0;

for i=1:K
    y_hat = y_hat + A(i)*normpdf(xp,mu(i),2);
end

% for j=1:N
%     lkh = lkh + log(normpdf(y(j)-y_hat(j),0,sigma_e));
% end
lkh = normpdf(y-y_hat,0,sigma_e);
lkh = sum(log(lkh));