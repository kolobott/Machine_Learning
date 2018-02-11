function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

for i = 1:size(z, 1)
    for j = 1:size(z, 2)
        g(i, j) = 1 / (1 + exp(-z(i, j)));
    end
end

%Compute the sigmoid of each value of z (z can be a matrix,
%vector or scalar).





% =============================================================

end
