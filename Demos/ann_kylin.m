%% This toolbox can be used as a benchmark test code 
%% Some basical test codes, to be continued...

% Artificial Neural Networks for Beginners
% http://blogs.mathworks.com/loren/2015/08/04/artificial-neural-networks-for-beginners//#view_comments
% rewritten and noted by Kylin 2016/8/15

%% Initialization
clear ; close all; clc 

% add the higher level folder path 
addpath('../data'); 
addpath('../util');
addpath('../NN');

original=2;

if original == 1 
load mnist_uint8;  % this dataset is total wrong!
end 

if original == 2
%% Load MNIST database files form yann website, this dataset is correct 
trainData   = loadMNISTImages('train-images.idx3-ubyte');
% trainData=trainData(:,1:6000); % sample 6000 samples 
trainLabels = loadMNISTLabels('train-labels.idx1-ubyte');
% trainLabels=trainLabels(1:6000); % the corresponding labels 
% mnistLabels(mnistLabels==0) = 10; % Remap 0 to 10     

testData   = loadMNISTImages('t10k-images.idx3-ubyte');
% testData=testData(:,1:1000);
testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
% testLabels=testLabels(1:1000);
% testLabels(testLabels==0) = 10; % Remap 0 to 10
end 

% cast = @double; % functional handle, for data type conversion 

% debug=1;
%% using 1/10 samples to test 
% if debug==1
%     train_x = train_x(1:6000,:);
%     test_x  = test_x(1:1000,:);
%     
%     train_y = train_y(1:6000,:);
%     test_y  = test_y(1:1000,:);
% end

% tr = csvread('train.csv', 1, 0);                  % read train.csv
% sub = csvread('test.csv', 1, 0);                  % read test.csv

%% Data visualization can be used to check samples
figure                                          % plot images
colormap(gray)                                  % set to grayscale
for i = 1:25                                    % preview first 25 samples
    subplot(5,5,i)                              % plot them in 6 x 6 grid
%   digit = reshape(train_x(i, 1:end), [28,28])';    % row = 28 x 28 image

    digit = reshape(trainData(1:end,i), [28,28]);       % row = 28 x 28 image
    imagesc(digit)                              % show the image
%   title(num2str(find(train_y(i, :)==1)))                    % show the label
    title(num2str(trainLabels(i, 1)))                    % show the label
end

display('Samples pass the check...!');  

%% Data Preparation (one-hot coding for data label)
% You will be using the nprtool pattern recognition app from Neural Network Toolbox. The app expects two sets of data:
% 
% inputs - a numeric matrix, each column representing the samples and rows the features. This is the scanned images of handwritten digits.
% targets - a numeric matrix of 0 and 1 that maps to specific labels that images represent. This is also known as a dummy variable. Neural Network Toolbox also expects labels stored in columns, rather than in rows.

% The dataset stores samples in rows rather than in columns, so you need to transpose it. Then you will partition the
% data so that you hold out 1/3 of the data for model evaluation, and you will only use 2/3 for training 
% our artificial neural network model.

n = size(trainData, 2);               % number of samples in the dataset
% targets  = tr(:,1);                 % 1st column is |label| 

% targets(targets == 0) = 10;         % use '10' to present '0'
trainLabels(trainLabels==0) = 10;     % Remap 0 to 10
testLabels(testLabels==0) = 10;       % Remap 0 to 10

% Dummy variable coding.
trainLabels = dummyvar(trainLabels);       % convert label into a dummy variable �Ʊ���
testLabels = dummyvar(testLabels);   

trainLabels=trainLabels';
testLabels=testLabels';

% inputs = tr(:,2:end);               % the rest of columns are predictors
% inputs = inputs';                   % transpose input ת�ú�Ϊ784*10000����
% targets = targets';                 % transpose target ���Ҳ��Ҫת��?10*10000 ����
% targetsd = targetsd';               % transpose dummy variable

rng(1);                             % for reproducibility ��������ÿ��ظ�?
c = cvpartition(n,'Holdout',n/3);   % hold out 1/3 of the dataset

Xtrain = trainData(:, training(c));      % 2/3 of the input for training
Ytrain = trainLabels(:, training(c));    % 2/3 of the target for training
Xtest = trainData(:, test(c));           % 1/3 of the input for testing
% Ytest = targets(test(c));              % 1/3 of the target for testing

Ytestd = trainLabels(:, test(c));        % 1/3 of the dummy variable for testing

% click the Pattern Recognition Tool to open the Neural Network Pattern Recognition Tool. 
% nnstart

 nprtool 

load myWeights                          % load the learned weights
W1 =zeros(100, 28*28);                  % pre-allocation
W1(:, x1_step1_keep) = IW1_1;           % reconstruct the full matrix
figure                                  % plot images
colormap(gray)                          % set to grayscale
for i = 1:25                            % preview first 25 samples
    subplot(5,5,i)                      % plot them in 6 x 6 grid
    digit = reshape(W1(i,:), [28,28])'; % row = 28 x 28 image
    imagesc(digit)                      % show the image
end

% Ypred = myNNfun(Xtest);             % predicts probability for each label
% Ypred(:, 1:5)                       % display the first 5 columns
% [~, Ypred] = max(Ypred);            % find the indices of max probabilities
% sum(Ytest == Ypred) / length(Ytest) % compare the predicted vs. actual

