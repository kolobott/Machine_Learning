function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
sum = 0;
for i = 1:m
  sum = sum + (theta' * X(i, :)' - y(i))^2;
endfor
J = sum / (2 * m);



% =========================================================================

end
