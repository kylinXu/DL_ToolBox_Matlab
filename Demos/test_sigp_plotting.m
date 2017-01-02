function test_sigp_plotting()
%%TEST_SIGP_PLOTTING test plotting functions
% Test plotting function created for signalP dataset

%% add paths to tool box and load data
close all
try
    gpuDevice
    gpu = 1;
catch
    gpu = 0;
end


cast = @double; 
n = 4;
[train_x,train_y,test_x,test_y] = n_output_data(n,cast);


%% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

%% ex1 vanilla neural net
rng(0);
nn                          = nnsetup([784 200 200 n]);

% load weights from trained network. The confusion matrix has zero
% entries when the neural network performs poorly. Load weights
% trained for 50 epochs to get nice looking plots. Zero entires might
% produce NaN's chen MCC, specificity and precision is calculated.

temp = load('testNN_fouroutput');   
nn = temp.nn;


nn.output                   = 'softmax';
nn.activation_function      = 'sigm';

nn.weightMaxL2norm          = 15;
nn.cast                     = @double;
nn.caststr                  = 'double';

nn.errfun                   = @nnsigp;  %  sets the error function that is run after each iteration
opts.numepochs              =  3;      %  Number of full sweeps through data
opts.momentum_variable      = [linspace(0.5,0.99,opts.numepochs)];
opts.learningRate_variable  =  2.*(linspace(0.998,0.5,opts.numepochs ));
opts.learningRate_variable  = opts.learningRate_variable.*opts.momentum_variable;
opts.plot                   = 1;
opts.batchsize              = 1000;         %  Take a mean gradient step over this many samples
opts.ntrainforeval          = 5000;         % only GPU:  number of training samples that are copied to the gpu and used to
opts.plotfun                = @nnplotsigp;  % sets the plotting function
opts.outputfolder           = 'testout/testplots';
% load a trained network  - Same settings as above

tt = tic;

opts.plotfun = @nnplottest;
nn.errfun    = @nntest;
[~,L,loss] = nntrain(nn, train_x, train_y, opts,test_x,test_y);  % cpu
[~,L,loss] = nntrain(nn, train_x, train_y, opts);
if gpu == 1
[~,L,loss] = nntrain_gpu(nn, train_x, train_y, opts,test_x,test_y);
[~,L,loss] = nntrain_gpu(nn, train_x, train_y, opts);
end


opts.plotfun = @nnplotmatthew;
nn.errfun    = @nnmatthew;
[~,L,loss] = nntrain(nn, train_x, train_y, opts,test_x,test_y);  % cpu
[~,L,loss] = nntrain(nn, train_x, train_y, opts);
if gpu == 1
[~,L,loss] = nntrain_gpu(nn, train_x, train_y, opts,test_x,test_y);
[~,L,loss] = nntrain_gpu(nn, train_x, train_y, opts);
end

opts.plotfun = @nnupdatefigures;
nn.errfun    = [];
[~,L,loss] = nntrain(nn, train_x, train_y, opts,test_x,test_y);  % cpu
[~,L,loss] = nntrain(nn, train_x, train_y, opts);
if gpu == 1
[~,L,loss] = nntrain_gpu(nn, train_x, train_y, opts,test_x,test_y);
[~,L,loss] = nntrain_gpu(nn, train_x, train_y, opts);
end

%[nn_gpu,L,loss] = nntrain_gpu(nn, train_x, train_y, opts); % GPU
toc(tt);
%[nn_cpu,L,loss] = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
fprintf('Error GPU : %f \n',er);

% 
    function [trainx,trainy,testx,testy] = n_output_data(n,cast)
        %% extract a datasize with n digitst from mnist;
        assert(n<11,'N is too big ')
        addpath('../data');
        addpath('../util');
        addpath('../NN');
        d = load('mnist_uint8');
        d.train_x = cast(d.train_x) / 255;
        d.test_x  = cast(d.test_x)  / 255;
        d.train_y = cast(d.train_y);
        d.test_y  = cast(d.test_y);
        
        trainx = []; 
        trainy = []; 
        testx = []; 
        testy = []; 
        

        
        
        for i=1:n
            trainx = [trainx; d.train_x(d.train_y(:,i) == 1,:)];
            trainy = [trainy; d.train_y(d.train_y(:,i) == 1,1:n)];
            
            testx = [testx; d.test_x(d.test_y(:,i) == 1,:)];
            testy = [testy; d.test_y(d.test_y(:,i) == 1,1:n)];
        end
    end
end




%% Test of n_output_data function
% strain  = [5923 12665 18623  24754  30596 36017 41935  48200 54051  60000];
% stest  = [980 2115 3147 4157 5139 6031 6989 8017 8991 10000];
% for n = 1:10
%   disp(n);
%   [train_x,train_y,test_x,test_y] = n_output_data(n,cast);
%   assert(size(train_x,1) == strain(n),'1'); assert(size(train_y,1) == strain(n),'2');
%   assert(size(test_x,1) == stest(n),'3'); assert(size(test_y,1) == stest(n),'4');
% end
