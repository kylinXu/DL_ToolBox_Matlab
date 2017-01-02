clear all


load mnist_uint8;



val_x = double(train_x) / 255;
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;

val_y = double(train_y);
train_y = double(train_y);
test_y  = double(test_y);

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

%% NN creation parameters:
%  default values given, set parameter manually only if you want to change it
%  set paramters AFTER calling nnsetup()

nn.activation_function              = 'tanh_opt';   %  Activation functions of hidden layers: 
                                                    % 'sigm' (sigmoid), 'tanh_opt' (optimal tanh), 'ReLU' (rectified linear)
nn.learningRate                     = 2;            %  learning rate 
                                                    % Note: typically needs to be lower when using 'sigm'
                                                    % activation function and non-normalized inputs.
nn.momentum                         = 0.5;          %  Momentum
nn.weightPenaltyL2                  = 0;            %  L2 regularization
nn.weightMaxL2norm                  = 0;            %  Max L2 norm of incoming weights to individual Neurons - see Hinton 2009 dropout paper            
nn.nonSparsityPenalty               = 0;            %  Non sparsity penalty
nn.sparsityTarget                   = 0.05;         %  Sparsity target (used only if SpaisityPenalty ~= 0)
nn.inputZeroMaskedFraction          = 0;            %  Used for Denoising AutoEncoders
nn.dropoutFraction                  = 0;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
nn.output                           = 'sigm';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'
nn.errfun                           = [];           %  Empty for standard error options: @nnmatthew, @nnmatthew_gpu
                                                    % @nntest is used for @nnplotnntest
                                                    % @nnsigp is used for SignalP networks

%% NN training paramters (opts):
%  default values, may be set by line:
%  opts = nnopts_setup; 

opts.validation             = 1;    % Is overruled by the number of arguments in nntrain - 
                                    % ie. nntrain must have 6 input arguments for opts.validation = 1
opts.plot                   = 0;    % Plots the training progress if set
opts.plotfun                = @nnupdatefigures; 
                                    % Plots network error, alternatives:
                                    % @nnplotmatthew (plots error and matthew coefficients for each class)
                                    % @nnplotnntest (plots error and misclassification rate)
                                    % @nnplotsigp (used with SignalP networks)
opts.outputfolder           = '';   % If set the network is saved to the path specified by outpufolder after every 100 epochs. 
                                    % If plot is enabled the figure is also saved here after every 10 epochs.
opts.learningRate_variable  = [];   % If set specifies a momentum for every epoch. 
                                    % ie length(opts.momentum) == opts.numepochs.
opts.momentum_variable      = [];   % If set specifies a learning rate for every epoch.
                                    % ie length(opts.learningRate_variable) == opts.numepochs.
opts.numepochs              = 1;    % Number of epochs (runs through the complete training data set)
opts.batchsize              = 100;  % Number of traning examples to average gradient over (one mini-batch size)
                                    % (set to size(train_x,1) to perform full-batch learning)
opts.ntrainforeval          = [];   % Only relevant for GPU training. Sets the number of evaluation training datasets to use. 
                                    % Set this parameter to something small if you run into memory problems

%% ex1 vanilla neural net
rng(0);
nn = nnsetup([784 100 10]);
nn.weightMaxL2norm                  = 15;            %  Max L2 norm of incoming weights to individual Neurons - see Hinton 2009 dropout paper            
opts = nnopts_setup;      % Default training options
opts.learningRate_variable = [linspace(10,0.01,20)];
opts.momentum_variable = [linspace(0.5,0.99,20)];
opts.plot = 1;
opts.numepochs =  20;   %  Number of full sweeps through data
opts.batchsize = 100;  %  Take a mean gradient step over this many samples

[nn,L,loss] = nntrain(nn, train_x, train_y, opts, val_x,val_y);

[er, bad] = nntest(nn, test_x, test_y);

assert(er < 0.1, 'Too big error');

% Make an artificial one and verify that we can predict it
x = zeros(1,28,28);
x(:, 14:15, 6:22) = 1;
x = reshape(x,1,28^2);
figure; visualize(x');
predicted = nnpredict(nn,x)-1;

assert(predicted == 1);
%% ex2 neural net with L2 weight decay
rng(0);
nn = nnsetup([784 100 10]);

nn.weightPenaltyL2 = 1e-4;  %  L2 weight decay
opts = nnopts_setup;
opts.numepochs =  1;        %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples


[nn,L,loss] = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
assert(er < 0.1, 'Too big error');


%% ex3 neural net with dropout
rng(0);
nn = nnsetup([784 100 10]);

nn.dropoutFraction = 0.5;   %  Dropout fraction 
opts.numepochs =  1;        %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples

[nn,L,loss] = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
assert(er < 0.1, 'Too big error');

%% ex4 neural net with sigmoid activation function
rng(0);
nn = nnsetup([784 100 10]);

nn.activation_function = 'sigm';    %  Sigmoid activation function
nn.learningRate = 1;                %  Sigm require a lower learning rate
opts.numepochs =  1;                %  Number of full sweeps through data
opts.batchsize = 100;               %  Take a mean gradient step over this many samples

[nn,L,loss] = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
assert(er < 0.1, 'Too big error');

%% ex5 plotting functionality
rng(0);
nn = nnsetup([784 20 10]);
opts.numepochs         = 5;            %  Number of full sweeps through data
nn.output              = 'softmax';    %  use softmax output
opts.batchsize         = 1000;         %  Take a mean gradient step over this many samples
opts.plot              = 1;            %  enable plotting

[nn,L,loss] = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
assert(er < 0.1, 'Too big error');

%% ex6 neural net with sigmoid activation and plotting of validation and training error
% split training data into training and validation data
vx   = train_x(1:10000,:);
tx = train_x(10001:end,:);
vy   = train_y(1:10000,:);
ty = train_y(10001:end,:);

%%
rng(0);
nn.activation_function  = 'sigm';
nn                      = nnsetup([784 200 10]);     
nn.output               = 'softmax';                   %  use softmax output
nn.errfun               = @nnmatthew;     
nn.weightPenaltyL2      = 1e-4;
opts.numepochs          = 30;  
opts.learningRate_variable = ones(1,opts.numepochs);
opts.momentum_variable     = 0.5*ones(1,opts.numepochs);
opts.numepochs          = 30;                           %  Number of full sweeps through data
opts.batchsize          = 1000;                        %  Take a mean gradient step over this many samples

opts.plot               = 1;                           %  enable plotting

%the default for errfun is nntest, the default for plotfun is updatefigures
                 %  This function is applied to train and optionally validation set should be format [er, notUsed] = name(nn, x, y)
opts.plotfun                = @nnplotmatthew;

[nn,L,loss] = nntrain(nn, tx, ty, opts, vx, vy);                %  nntrain takes validation set as last two arguments (optionally)


[er, bad] = nntest(nn, test_x, test_y);
assert(er < 0.1, 'Too big error');