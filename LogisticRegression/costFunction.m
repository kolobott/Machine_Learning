function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

J = ((-y)'*log(sigmoid(X*theta)) - (1 - y)'*log(1 - sigmoid(X*theta)))/m;
for i = 1:size(theta, 1)
    grad(i, 1) = ((sigmoid(X*theta) - y)'*X(:, i))/m;
end
%Compute the cost of a particular choice of theta.
%Compute the partial derivatives and set grad to the partial
%derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta



% =============================================================

end
