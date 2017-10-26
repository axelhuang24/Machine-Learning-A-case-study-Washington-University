function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. 

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1); % m = 5000
         

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


class_y = zeros(m, num_labels); % 5000 * 10
for i = 1:num_labels
    class_y(:,i) = (y==i);
end

a1 = [ones(m,1),X]; % 5000 * 401
z2 = a1 * Theta1'; % 5000 * 25
a2 = sigmoid(z2); % 5000 * 25
a2 = [ones(m,1),a2]; % 5000 * 26
z3 = a2 * Theta2'; % 5000 * 10
a3 = sigmoid(z3); % 5000 * 10
h = a3; % 5000 * 10

regularization = 0;
J = -1/m*sum(sum(class_y .* log(h) + (1-class_y) .* log(1-h))); % cost function without regularization
regularization = lambda/(2*m) * ( sum(sum( Theta1(:, 2:end).^2 )) + sum(sum(Theta2(:, 2:end).^2 )) );
J = J + regularization;

% Backpropagation algorithm
delta3 = h - class_y; % 5000 * 10
delta2 = (delta3 * Theta2) .* (a2 .* (1-a2)); % 5000 * 26
delta2 = delta2(:,2:end); % 5000 * 25

Delta2 = delta3' * a2; % 10 * 26
Delta1 = delta2' * a1; % 25 * 401

Delta2_reg = Delta2 + lambda * [zeros(size(Theta2, 1), 1) Theta2( : , 2 : end)]; 
Delta1_reg = Delta1 + lambda * [zeros(size(Theta1, 1), 1) Theta1( : , 2 : end)]; 

Theta2_grad = Delta2_reg / m;
Theta1_grad = Delta1_reg / m;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
