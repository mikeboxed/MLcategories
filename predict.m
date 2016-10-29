function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%layer 1 and 2
X = [ones(m, 1) X];
a = sigmoid(X * Theta1'); 
a = [ones(m, 1) a];
disp(size(a));

a2 = sigmoid(a * Theta2');
disp('a2')
disp(size(a2))
[max_element, max_index]=  max(a2, [], 2);
p = max_index;

%prob_temp = (X * all_theta'); 
%disp(prob_temp)
%prob_temp = sigmoid(prob_temp);
%disp(prob_temp)
%[max_element, max_index]=  max(prob_temp, [], 2);
%p = max_index;


%disp(a)
%prob_temp = sigmoid(prob_temp);
%disp(prob_temp)
%[max_element, max_index]=  max(prob_temp, [], 2);
%p = max_index;

%GET FEAUTRES
%prob_temp = (X * all_theta'); 
%disp(prob_temp)
%prob_temp = sigmoid(prob_temp);
%disp(prob_temp)
%[max_element, max_index]=  max(prob_temp, [], 2);
%p = max_index;



% =========================================================================


end
