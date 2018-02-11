function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
  sum1 = 0;
  sum = 0;
  for i = 1:m
    sum = sum + (theta' * X(i, :)' - y(i));
  end
  sum = sum / m;
  temp_1 = theta(1) - alpha * sum;
  for i = 1:m
    sum1 = sum1 + (theta' * X(i, :)' - y(i))*X(i, 2);
  end
  sum1 = sum1 / m;
  temp_2 = theta(2) - alpha * sum1;
  theta(1) = temp_1;
  theta(2) = temp_2;


  % Save the cost J in every iteration    
  J_history(iter) = computeCost(X, y, theta);

end

end
