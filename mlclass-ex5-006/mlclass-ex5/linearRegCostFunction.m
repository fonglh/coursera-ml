function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% normal cost function
H = X * theta;
diff = H-y;
squares = diff .^ 2;
J = sum(squares) / (2*m);

% add regularization
J = J + lambda/(2*m) * sum(theta .^ 2)
% subtract away theta(1) which should be be regularized
J = J - lambda/(2*m) * theta(1) * theta(1)

grad = (X' * (H-y)) / m + lambda/m * theta;
% set gradient w.r.t theta(1) to a value without the regularization parameter
grad(1) = ((X' * (H-y))/m)(1);










% =========================================================================

grad = grad(:);

end
