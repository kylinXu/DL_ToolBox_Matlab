function test_example_NN_gpu()
%% TEST_EXAMPLE_NN_GPU
% demonstrates performance on mnist dataset
% uses
%   -   L2 norm weight constraint, see hinton 2012 (http://arxiv.org/pdf/1207.0580.pdf)
%   -   Dropout 50 %
%   -   Dropout of input 20 %
%   -   Increasing  Momentum 
%   -   Decreasing momentum
%
% Also demonstrates the plotting functionality:
% You can create your own error functions and plot then.
% Error function must be of the format: [er, bad] = funname(nn, x, y)
%           - bad is a dummy included, set it to []
%           - err is  1 X D ROW vector of the calculated errors
%
% The plotting functino has the format:
%     funname(nn,fhandle,L,opts,i)
%       i       : number of current epoch
%       fhandle : handle to the plotting figure
%       L       : loss struct where
%                   * L.train.e [i x D] vector of taining set errors
%                   * L.val.e [i x D] vector of val set errors (optional, if val data is supplied)
%                   * L.train.e_errfun [i x D] vector of taining set errors from
%                       supplied error function
%                   * L.val.e_errfun [i x D] vector of val set errors from
%                       supplied error function  (Optional if val data is supplied)
%
% Note that the networks produced these files are not compatible with the
% original deeplearning toolbox including the RBM's because of w,b
% notation. To use RBM's use the original nnsetup function and afterwards: 
%  add b to w like nnRBM.w{1} = [nn.b{1} nn.W{w}] for all layers
addpath('../data');
addpath('../util');
addpath('../NN');

%% Load dataset
% Load MNIST database files form yann lecun 's website, this dataset is correct 
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

%% Data visualization can be used to check samples
figure                                          % plot images
colormap(gray)                                  % set to grayscale
for i = 1:25                                    % preview first 25 samples
    subplot(5,5,i)                              % plot them in 6 x 6 grid
%   digit = reshape(train_x(i, 1:end), [28,28])';       % row = 28 x 28 image

    digit = reshape(trainData(1:end,i), [28,28]);       % row = 28 x 28 image
    imagesc(digit)                                      % show the image
%   title(num2str(find(train_y(i, :)==1)))              % show the label
    title(num2str(trainLabels(i, 1)))                   % show the label
end
display('Samples pass the check...!');  

%% Data Preparation
% You will be using the nprtool pattern recognition app from Neural Network Toolbox. The app expects two sets of data:
% 
% inputs - a numeric matrix, each column representing the samples and rows the features. This is the scanned images of handwritten digits.
% targets - a numeric matrix of 0 and 1 that maps to specific labels that images represent. This is also known as a dummy variable.
% Neural Network Toolbox also expects labels stored in columns, rather than in rows.

% The dataset stores samples in rows rather than in columns, so you need to transpose it. Then you will partition the
% data so that you hold out 1/3 of the data for model evaluation, and you will only use 2/3 for training 
% our artificial neural network model.

% n = size(trainData, 2);             % number of samples in the dataset
% targets  = tr(:,1);                 % 1st column is |label|

% targets(targets == 0) = 10;         % use '10' to present '0'
trainLabels(trainLabels==0) = 10;     % Remap 0 to 10
testLabels(testLabels==0) = 10;       % Remap 0 to 10

% Dummy variable coding. (one-hot coding for label)
trainLabels = dummyvar(trainLabels);       % convert label into a dummy variable
testLabels = dummyvar(testLabels);   

% Dajust the data form 
train_x=trainData';
test_x=testData';

train_y=trainLabels;
test_y=testLabels;

clear trainData testData trainLabels testLabels

%% Using one in ten of amples for test code 
debug=0;
if debug==1
    train_x = train_x(1:6000,:);
    test_x  = test_x(1:1000,:);
    train_y = train_y(1:6000,:);
    test_y  = test_y(1:1000,:);
end


%% Simple rescale the feature of samples to (0,1) and type conversion (uint to double)
% train_x = double(train_x) / 255;
% test_x  = double(test_x)  / 255;
% 
% train_y = double(train_y); 
% test_y  = double(test_y);

cast = @double; %  function handle
train_x = cast(train_x) / 255;
test_x  = cast(test_x)  / 255;
train_y = cast(train_y);
test_y  = cast(test_y);

% The size of training dataset
% dataSize=length(train_x);

%% Standard normalize (zero Mean and unit Standard deviation)  
% [Z,mu,sigma] = zscore(___) also returns the means and standard deviations
% used for centering and scaling, mu and sigma, respectively.
% The columns of X are centered to have mean 0 and scaled to have standard deviation 1.
% Z is the same size as X.

[train_x, mu, sigma] = zscore(train_x); % Feature standardization dealing with each column
test_x = normalize(test_x, mu, sigma);  % Using the mu and sigma on the training set to normalize the test set for the same treatment


rng(0);
% nn                        = nnsetup([784 1200 1200 1200 10]);
nn                          = nnsetup([784 800 800 10]);
nn.output                   = 'softmax'; % output function: softmax | sigm | linear
nn.activation_function      = 'sigm';    % activation func: sigm | tanh_opt | linear
nn.dropoutFraction          = 0.5;       % Droupout of hidden layers
nn.inputZeroMaskedFraction  = 0.2;       % input dropout
%nn.weightPenaltyL2         = 1e-6;      % weightdecay
nn.weightMaxL2norm          = 15;        % L" norm of incoming weights to each neuron are constrined to be below this value, rescaled if above
nn.cast                     = @double;   % double or single precision, single cuts memory usage by app. 50%
nn.caststr                  = 'double';  % double or single precision, single cuts memory usage by app. 50%
nn.errfun                   = @nntest;

opts.plotfun                = @nnplottest;
opts.numepochs              =  5000;        %  Number of full sweeps through data
opts.momentum_variable      = [linspace(0.5,0.95,1500 ) linspace(0.95,0.95,opts.numepochs -1500)];
opts.learningRate_variable  =  8.*(linspace(0.998,0.998,opts.numepochs ).^linspace(1,opts.numepochs,opts.numepochs ));
opts.learningRate_variable  = opts.learningRate_variable.*opts.momentum_variable;
opts.plot                   = 1;            % 0 = no plotting, migth speed up calc if epochs run fast
opts.batchsize              = 1000;         % Take a mean gradient step over this many samples. GPU note: below 500 is slow on GPU because of memory transfer
opts.ntrainforeval          = 5000;         % number of training samples that are copied to the gpu and used to evalute training performance
opts.outputfolder           = 'nns/hinton'; % saves network each 100 epochs and figures after 10. hinton is prefix to the files. 
                                            % nns is the name of a folder
                                            % from where this script is
                                            % called (probably tests/nns)
                                        
tt = tic; 
[nn,L,loss]                 = nntrain_gpu(nn, train_x, train_y, opts,test_x,test_y); %use nntrain to train on cpu
toc(tt);
[er_gpu, bad]               = nntest(nn, test_x, test_y);    
fprintf('Error: %f \n',er_gpu);
