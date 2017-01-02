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

%mΪѵ��������������e.g.,6000��
m = size(train_x,1);
assert(m ~= 0)

% check if error function is supplied, else use nntest
if isempty(nn.errfun)
    nn.errfun = @nntest;
end;

% determine number of returned error values ��һ����������������Ԥ���size
nerrfun =  numel(nn.errfun(nn, train_x(1,:), train_y(1,:)));  

% ��ʼ��ѵ������֤���Ľṹ�� 
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

%% ����opt����
% mini-batch��Ĵ�С
batchsize = opts.batchsize;
%ɨ��ȫ�������Ĵ�������ʵ��5�У�ɨ��������5�Σ�
numepochs = opts.numepochs;
% mini-batch��ĸ����� (i.e. ex5��1000)
numbatches = floor(m / batchsize);
% LΪbatch������ɨ�����������ĳ˻� �� ex5,300*1
L = zeros(numepochs*numbatches,1);
n = 1;
for i = 1 : numepochs
    % ��ʱ
    epochtime = tic;
    % update momentum 
    if var_momentum_flag
        nn.momentum = opts.momentum_variable(i);
    end
    % update learning rate 
    if var_learningRate_flag
        nn.learningRate = opts.learningRate_variable(i);
    end
    % ����һ��������������С���������kk��i.e.��1*6000��
    % ����ʵ����ȫ�������������ȡmini-batch size��С������
    kk = randperm(m);
    for l = 1 : numbatches  % ��ÿ��batch����ѵ��
        % ���ȡ��һ��mini-batch��С��ѵ������
        batch_x = extractminibatch(kk,l,batchsize,train_x);    
        %Add noise to input (for use in denoising autoencoder)
        % ��������dropout ��e.g., 0.2��
        if(nn.inputZeroMaskedFraction ~= 0)
            % ��������С��nn.inputZeroMaskedFractionˮƽ��Ԫ����Ϊ0
            % ��������������������Ҫ������Ԥ����[0,1]��
            batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
        end
        % ��ȡmini-batch��С��������ǩ
        batch_y = extractminibatch(kk,l,batchsize,train_y);
        % ǰ�򴫲�
        % NNFF performs a feedforward pass
        % nn = nnff(nn, x, y) returns an neural network structure with updated
        % layer activations, error and loss (nn.a, nn.e and nn.L)
        nn = nnff(nn, batch_x, batch_y);
        % ����
        % NNBP performs backpropagation
        % nn = nnbp(nn) returns an neural network structure with updated weights
        nn = nnbp(nn);
        % �ݶ��½���SGD��
        % NNAPPLYGRADS updates weights and biases with calculated gradients
        % nn = nnapplygrads(nn) returns an neural network structure with updated
        % weights and biases
        nn = nnapplygrads(nn);
        % ÿһ��Ԫ�ر�ʾһ��mini-batch�ݶ��½�һ�εõ������ֵ
        L(n) = nn.L; % 600��batch��ɨ�裨������һ�Σ�LΪÿһ��batch��������ʧ����ֵ
        n = n + 1;
    end
    
    t1 = toc(epochtime);   
    evalt = tic;  
    %after each epoch update losses
    % ��֤ѡ��
    if opts.validation == 1
        loss =  nneval(nn, loss,i,train_x, train_y, val_x, val_y);     
        % û���ϣ�
%         str_perf = sprintf('; Full-batch train mse = %f, val mse = %f', loss.train.e(end), loss.val.e(end));
%         switch nn.output
%             case {'sigm', 'linear'}
%                 str_perf = sprintf('; Full-batch train MSE = %f, val mse = %f', loss.train.e(end), loss.val.e(end));
%             case 'softmax'
%                 str_perf = sprintf('; Full-batch train MCE = %f, val mse = %f', loss.train.e(end), loss.val.e(end));
%         end
    else
        loss = nneval(nn, loss,i,train_x, train_y);% ��Full-batch train�����õ�һ�����
%         str_perf = sprintf('; Full-batch train err = %f', loss.train.e(end));
    end
    
    % ��ͼ
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

