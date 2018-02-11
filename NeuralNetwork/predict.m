function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);
X = [ones(m,1) X];
hidden_layer = zeros(size(Theta2, 2)-1, size(X, 1));
hidden_layer = sigmoid(Theta1*X');
hidden_layer = [ones(1, m); hidden_layer];
pred = sigmoid(hidden_layer'*Theta2');
[maxs, p] = max(pred, [], 2);










% =========================================================================


end
