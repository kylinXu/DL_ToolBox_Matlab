%% Genome completeness: A novel approach using core genes 
%% the dataset may be accessed in http://korflab.ucdavis.edu/datasets/genome_completeness/index.html#SCT3 

addpath('../../dataanalysis/')
addpath('../data');
addpath('../util');
addpath('../NN');
p = matfile('Eukaryota_sequence_homepage_data_windowsize41.mat');

train_x = p.Eukar_input1(1:20000,:);
train_y = p.Eukar_target1(1:20000,:);
val_x = p.Eukar_input1(20001:25000,:);
val_y = p.Eukar_target1(20001:25000,:);

m = size(train_x,2);
t = size(train_y,2);


%% ex1 vanilla neural net
rng(0);
nn                          = nnsetup([m 100 50 t]);
nn.output                   = 'softmax';
nn.activation_function      = 'sigm';
nn.normalize_input          = 0;
nn.dropoutFraction          = 0.5;
nn.inputZeroMaskedFraction  = 0.2;
nn.weightPenaltyL2          = 1e-6;
%nn.weightMaxL2norm = 15;
nn.cast                     = @double;
nn.caststr                  = 'double';
nn.errfun                   = @nnsigp;% @nnsigp;
opts.plotfun                = @nnplotsigp; %@nnplotsigp;
opts.numepochs              = 100;   %  Number of full sweeps through data
%opts.momentum_variable      = [linspace(0.5,0.99,opts.numepochs/2) linspace(0.99,0.99,opts.numepochs/2)];
%opts.learningRate_variable  = 2.*(linspace(0.998,0.998,opts.numepochs).^linspace(1,998,opts.numepochs));
%opts.learningRate_variable  = opts.learningRate_variable.*opts.momentum_variable;
opts.plot                   = 1;
opts.batchsize = 100;  %  Take a mean gradient step over this many samples
opts.ntrainforeval = 5000; % number of training samples that are copied to the gpu and used to 
                           % evalute training performance
                           % if you have a small dataset set this to number
                           % of samples in your training data
tt = tic;
                           [nn_gpu,L,loss] = nntrain(nn, train_x, train_y, opts, val_x, val_y);
toc(tt);
                           %[nn_cpu,L,loss] = nntrain(nn, train_x, train_y, opts);
%[er_gpu, bad] = nntest(nn_gpu, test_x, test_y);
%[er_cpu, bad] = nntest(nn_cpu, test_x, test_y);
fprintf('Error GPU (single): %f \n',er_gpu);
%fprintf('Error GPU (single); %f \n',er_cpu);
