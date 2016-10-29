function [p,rate] = topcatlist(all_theta, X,y,q)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

% q = number of categories to look at, 
%    ie 1 = only test for most purchased category, 4 = try to estimate top 4 categories

m = size(X, 1);
num_labels = size(all_theta, 1);
%default value is to only look at top category
% You need to return the following variables correctly 
p = zeros(m, num_labels);
pred_topcat = zeros(m, num_labels); %creates empty gid to house locations of top 
% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).top
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       
rate = 0;
prob_temp = (X * all_theta'); 
prob_temp = sigmoid(prob_temp);
%disp('topcatlist prob_temp')
%disp(prob_temp);
%pause;

for c = 1:m
    [temp_value, temp_index] = sort(prob_temp(c,:),'descend'); %reorders for highest probabilities
    temp_topcats_max = sort(temp_value(q)); %only looks at values greater than top 3 or whatever q is
    %disp('topcatlist temp_topcats_max')
    %disp(temp_topcats_max);
    %pause;
    pred_topcat(c,:)= (prob_temp(c,:)>= temp_topcats_max); %only keep probabilities if probabilit of that categoryis greater tahn top 3rd cate and change to 1,everythign else stays 0
    %disp('pred_topcat')
    %disp(pred_topcat);
    %pause;
end

total_pred_count = sum(pred_topcat(:)); %total number of all examples of top categories

pred_topcat_temp = (pred_topcat==1 & pred_topcat==y); % matrix of where predict matches Y output, for top 3 (q) categories 
pred_count_correct_temp = sum(pred_topcat_temp(:));

%disp('accuracy rate for top categories only');
rate = (pred_count_correct_temp / total_pred_count); % percentage of top categories that matches y
p = pred_topcat; %matrix of predicted top 3 (q) categories
pause;



% =========================================================================


end
