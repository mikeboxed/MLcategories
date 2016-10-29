%% Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%     lrCostFunction.m (logistic regression cost function)
%     oneVsAll.m

%     predictOneVsAll.m
%     predict.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 11;  % orig= 400; 20x20 Input Images of Digits
num_labels = 11;          % orig10; 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
threshold = 0.50;   %threshold to mark as accurately assigned(0-1)
c = 1000; %number of examples to take
q = 4; %number of top categories to look at - default value 1, top category only

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('categoryupload.mat'); % training data stored in arrays X, y
disp('Number of training examples; m=');
m = size(x([1:c],:), 1);%only take first 1000 examples
disp(m);
% Randomly select 100 data points to display
%rand_indices = randperm(m);
%sel = testexamplesX_usercat(rand_indices(1:100), :);

%displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============ Part 2: Vectorize Logistic Regression ============
%  In this part of the exercise, you will reuse your logistic regression
%  code from the last exercise. You task here is to make sure that your
%  regularized logistic regression implementation is vectorized. After
%  that, you will implement one-vs-all classification for the handwritten
%  digit dataset.
%

fprintf('\nTraining One-vs-All Logistic Regression...\n')

X = x([1:c],[1:input_layer_size]);
y = ycatbinary([1:c],:);

lambda = 1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);
fprintf('Thetas calculated; program paused. Press enter to continue.\n');
%disp(all_theta); %estimated optimized thetas
pause;

%% ================ Part 3: Predict for One-Vs-All ================
%  After ...
disp('Starting Predict for One-Vs-All:');
%pred = predictOneVsAll(all_theta, X);

[pred, rate] = predictcategory(all_theta, X, y,threshold);
disp('Accuracy rate (Total/All category match (paused, hit enter to continue:');
disp(rate)
pause; 

disp('accuracy rate for top categories only, for top q number of categories:');
[pred_top,rate_top] = topcatlist(all_theta, X,y,q);
disp('number of categories to looked at=')
disp(q)
disp(rate_top);
%fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
pause;

%%%%%cross validation%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nTraining Set Accuracy(cross validation): %f\n');
X_cv = x([1001:1335],[1:input_layer_size]);
y_cv = ycatbinary([1001:1335],:);
lambda = 1;
%[all_theta] = oneVsAll(X_cv, y_cv, num_labels, lambda);
pause; 

[pred_cv, rate_cv] = predictcategory(all_theta, X_cv, y_cv,threshold);
disp('Accuracy rate (CV) (Total/All category match (paused, hit enter to continue:');
disp(rate_cv)
pause; 

disp('Accuracy rate for top categories only (CV), for top q number of categories:');
disp('number of categories to looked at=')
disp(q)
[pred_top_cv,rate_top_cv] = topcatlist(all_theta, X_cv,y_cv,q);
disp(rate_top_cv);
pause;

%%%%%cross validation%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% =========== Part 5: Learning Curve for Linear Regression =============
%
%
%lambda = 0;
%[error_train, error_val] = ...
%    learningCurve([ones(m, 1) X], y, ...
%                  [ones(size(Xval, 1), 1) Xval], yval, ...
%                  lambda);
%
%plot(1:m, error_train, 1:m, error_val);
%title('Learning curve for linear regression')
%legend('Train', 'Cross Validation')
%xlabel('Number of training examples')
%ylabel('Error')
%axis([0 13 0 150])

%fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
%for i = 1:m
%    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
%end



%fprintf('Program paused. Press enter to continue.\n');
%pause;

%n <- length(x)
%sort(x,partial=n-1)[n-1]
