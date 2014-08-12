function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

values = [0.01 0.03 0.1 0.3 1 3 10 30];
predict_error_size = size(values, 2);
predict_error = zeros(predict_error_size);

for i=1:size(values, 2);
  for j=1:size(values, 2);
    model = svmTrain(X, y, values(i), @(x1, x2) gaussianKernel(x1, x2, values(j)));
    predictions = svmPredict(model, Xval);
    predict_error(i, j) = mean(double(predictions ~= yval));
  end
end

% gets the minimum in each column as a row vector
% row_index should contain the row numbers of the min element in each column
[col_min, row_index] = min(predict_error);
% gets the minimum in the row vector
% col_index indicates which column has the smallest item
[min_error, col_index] = min(col_min);

% row_index(col_index) is the row number of the smallest item
% col_index is the column number of the smallest item

C = values(row_index(col_index));
sigma = values(col_index);

% =========================================================================

end
