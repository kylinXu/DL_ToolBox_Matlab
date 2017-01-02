% %% 清空环境变量
% clc;
% clear;
% % close all;


function test_example_DBN
load mnist_uint8;
% load New_data

debug=1;
if debug==1
    % 只用十分之一的样本测试下代码
    train_x = train_x(1:6000,:);
    test_x  = test_x(1:1000,:);
    train_y = train_y(1:6000,:);
    test_y  = test_y(1:1000,:);
end

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

% %% DBN parameters:
% %  if default value is given, parameter may not be set in user code
% 
% opts.numepochs  = 10;       % number of epochs (full sweeps through data)
% opts.batchsize  = 100;      % number of traning examples to average gradient over (one mini-batch size)
%                             % (set to size(train_x,1) to perform full-batch learning)
% opts.momentum   = 0;        % learning momentum (default: 0)
% opts.alpha      = 1;        % learning rate
% opts.cdn        = 1;        % number of steps for contrastive divergence learning (default: 1)
% opts.vis_units  = 'sigm';   % type of visible units (default: 'sigm')
% opts.hid_units  = 'sigm';   % type of hidden units  (default: 'sigm')
%                             % units can be 'sigm' - sigmoid, 'linear' - linear
%                             % 'NReLU' - noisy rectified linear (Gaussian noise)
% dbn.sizes       = [10 20];  % size of hidden layers

%%  ex1 train a 100 hidden unit RBM and visualize its weights
% rng('default'),rng(0);
% dbn.sizes = [100];
% opts.numepochs =   1;
% opts.batchsize = 100;
% opts.momentum  =   0;
% opts.alpha     =   1;    
% opts.cdn       =   1;
% dbn = dbnsetup(dbn, train_x, opts);
% dbn = dbntrain(dbn, train_x, opts);
% figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

% Use code like this to visualize non-square images:
% X = dbn.rbm{1}.W';
% vert_size = 28;
% hor_size = 28;
% figure; visualize(X, [min(X(:)) max(X(:))], vert_size, hor_size);

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rand('state',0)
%train dbn
dbn.sizes = [100 100];
opts.numepochs =   1;
opts.batchsize = 100;
opts.momentum  =   0; % 动量
opts.alpha     =   1; % 学习率  
dbn = dbnsetup(dbn, train_x, opts);% 构建一个DBN
dbn = dbntrain(dbn, train_x, opts);% 训练网络
figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights
% figure; visualize(dbn.rbm{2}.W');   %  Visualize the RBM weights

%unfold dbn to nn   DBN的每一层训练完成后自然还要把参数传递给一个大的NN，这就是这个函数的作用
nn = dbnunfoldtonn(dbn, 10);
nn.activation_function = 'sigm';
% nn.output='softmax';

%% train nn
opts.numepochs =  1;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);  

assert(er < 0.10, 'Too big error'); 
