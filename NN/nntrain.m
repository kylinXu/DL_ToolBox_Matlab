function [nn, L,loss]  = nntrain(nn, train_x, train_y, opts, val_x, val_y)
%NNTRAIN trains a neural net on cpu
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.

assert(isfloat(train_x), 'train_x must be a float');
assert(nargin == 4 || nargin == 6,'number ofinput arguments must be 4 or 6')

nn.isGPU = 0; % tell code that variables are not on gpu

%m为训练样本的数量（e.g.,6000）
m = size(train_x,1);
assert(m ~= 0)

% check if error function is supplied, else use nntest
if isempty(nn.errfun)
    nn.errfun = @nntest;
end;

% determine number of returned error values 拿一个样本评估下网络预测的size
nerrfun =  numel(nn.errfun(nn, train_x(1,:), train_y(1,:)));  

% 初始化训练和验证误差的结构体 
loss.val.e_errfun          = zeros(opts.numepochs,nerrfun);
loss.train.e_errfun        = zeros(opts.numepochs,nerrfun);


loss.train.e               = zeros(opts.numepochs,1);
loss.val.e                 = zeros(opts.numepochs,1);

if nargin == 6
    opts.validation = 1;
else
    opts.validation = 0;
end

fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
    % check if plotting function is supplied, else use nnupdatefigures
    if ~isfield(opts,'plotfun') || isempty(opts.plot)
        opts.plotfun = @nnupdatefigures;
    end
    
end

if isfield(opts, 'outputfolder') && ~isempty(opts.outputfolder)
    save_nn_flag = 1;
else
    save_nn_flag = 0;
end

%variable momentum ?
if isfield(opts, 'momentum_variable') && ~isempty(opts.momentum_variable)
    if length(opts.momentum_variable) ~= opts.numepochs
        error('opts.momentum_variable must specify a momentum value for each epoch ie length(opts.momentum_variable) == opts.numepochs')
    end
    var_momentum_flag = 1;
else
    var_momentum_flag = 0;
end

%variable learningrate ?
if isfield(opts, 'learningRate_variable') && ~isempty(opts.learningRate_variable)
    if length(opts.learningRate_variable) ~= opts.numepochs
        error('opts.learningRate_variable must specify a learninrate value for each epoch ie length(opts.learningRate_variable) == opts.numepochs')
    end
    var_learningRate_flag = 1;
else
    var_learningRate_flag = 0;
end

%% 设置opt参数
% mini-batch块的大小
batchsize = opts.batchsize;
%扫描全部样本的次数，（实验5中，扫（迭代）5次）
numepochs = opts.numepochs;
% mini-batch块的个数。 (i.e. ex5：1000)
numbatches = floor(m / batchsize);
% L为batch数量和扫描样本次数的乘积 。 ex5,300*1
L = zeros(numepochs*numbatches,1);
n = 1;
for i = 1 : numepochs
    % 计时
    epochtime = tic;
    % update momentum 
    if var_momentum_flag
        nn.momentum = opts.momentum_variable(i);
    end
    % update learning rate 
    if var_learningRate_flag
        nn.learningRate = opts.learningRate_variable(i);
    end
    % 生成一个和整个样本大小的随机向量kk（i.e.，1*6000）
    % 这样实现在全部样本中随机抽取mini-batch size大小的样本
    kk = randperm(m);
    for l = 1 : numbatches  % 对每个batch进行训练
        % 随机取出一个mini-batch大小的训练样本
        batch_x = extractminibatch(kk,l,batchsize,train_x);    
        %Add noise to input (for use in denoising autoencoder)
        % 对输入做dropout （e.g., 0.2）
        if(nn.inputZeroMaskedFraction ~= 0)
            % 将样本中小于nn.inputZeroMaskedFraction水平的元素置为0
            % （这里这样做，首先需要将样本预处理到[0,1]）
            batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
        end
        % 抽取mini-batch大小的样本标签
        batch_y = extractminibatch(kk,l,batchsize,train_y);
        % 前向传播
        % NNFF performs a feedforward pass
        % nn = nnff(nn, x, y) returns an neural network structure with updated
        % layer activations, error and loss (nn.a, nn.e and nn.L)
        nn = nnff(nn, batch_x, batch_y);
        % 误差反传
        % NNBP performs backpropagation
        % nn = nnbp(nn) returns an neural network structure with updated weights
        nn = nnbp(nn);
        % 梯度下降（SGD）
        % NNAPPLYGRADS updates weights and biases with calculated gradients
        % nn = nnapplygrads(nn) returns an neural network structure with updated
        % weights and biases
        nn = nnapplygrads(nn);
        % 每一个元素表示一个mini-batch梯度下降一次得到的误差值
        L(n) = nn.L; % 600个batch，扫描（迭代）一次，L为每一次batch迭代的损失函数值
        n = n + 1;
    end
    
    t1 = toc(epochtime);   
    evalt = tic;  
    %after each epoch update losses
    % 验证选项
    if opts.validation == 1
        loss =  nneval(nn, loss,i,train_x, train_y, val_x, val_y);     
        % 没用上？
%         str_perf = sprintf('; Full-batch train mse = %f, val mse = %f', loss.train.e(end), loss.val.e(end));
%         switch nn.output
%             case {'sigm', 'linear'}
%                 str_perf = sprintf('; Full-batch train MSE = %f, val mse = %f', loss.train.e(end), loss.val.e(end));
%             case 'softmax'
%                 str_perf = sprintf('; Full-batch train MCE = %f, val mse = %f', loss.train.e(end), loss.val.e(end));
%         end
    else
        loss = nneval(nn, loss,i,train_x, train_y);% 用Full-batch train，来得到一个误差
%         str_perf = sprintf('; Full-batch train err = %f', loss.train.e(end));
    end
    
    % 画图
    % plot if figure is available
    if ishandle(fhandle)
        opts.plotfun(nn, fhandle, loss, opts, i);
        
        %save figure to the output folder after every 10 epochs
        if save_nn_flag && mod(i,10) == 0
            save_figure(fhandle,opts.outputfolder,2,[40 25],14);
            disp(['Saved figure to: ' opts.outputfolder]);
        end
    end
      
    t2 = toc(evalt);
    
    switch nn.output
        case {'sigm', 'linear'}
            disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  ...
                '. Took ' num2str(t1) ' seconds' '. MSE on training set is '...
                num2str(mean(L((n-numbatches):(n-1)))) '. Eval time: ' num2str(t2)...
                ' Learningrate: ' num2str(nn.learningRate) ' Momentum: ' num2str(nn.momentum)]);
        case 'softmax'
            disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  ...
                '. Took ' num2str(t1) ' seconds' '.MCE on training set is '...
                num2str(mean(L((n-numbatches):(n-1)))) '. Eval time: ' num2str(t2)...
                ' Learningrate: ' num2str(nn.learningRate) ' Momentum: ' num2str(nn.momentum)]);       
    end
              
        %save model after very 100 epochs
    if save_nn_flag && mod(i,100) == 0 
        epoch_nr = i;
        save([opts.outputfolder '_epochnr' num2str(epoch_nr) '.mat'],'nn','opts','epoch_nr','loss');
        disp(['Saved weights to: ' opts.outputfolder]);
    end
end
end

