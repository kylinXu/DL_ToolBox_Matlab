function nn = nnff(nn, x, y)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

n = nn.n; % n表示网络层数或网络深度
m = size(x, 1); % 训练时为mini-batch大小的样本，测试时为测试样本大小
% x = [ones(m,1) x]; % 这里是对输入X增加1列（bias）
nn.a{1} = x;       % 这里用a1表示输入x，也就是输入层的输出
cast = @nn.cast;
cstr = nn.caststr;

%feedforward pass
for i = 2 : n-1
    % 这里实现的记法：X是m*n维，m为样本个数，n为样本维数。 W是k*n维，k为隐层节点数。
    z = bsxfun(@plus, nn.a{i - 1} * nn.W{i - 1}',nn.b{i-1}'); %input to each layer
    switch nn.activation_function
        case 'sigm'
            % Calculate the unit's outputs (including the bias term)
            nn.a{i} = sigm(z);
        case 'tanh_opt'
            nn.a{i} = tanh_opt(z);
        case 'ReLU'  % linear rectified units max(0,x)
            nn.a{i} = ReLU(z);
            %             nn.a{i} = max(0 ,z);
        case 'AReLU'
            % 绝对值大于0的
            nn.a{i} = AReLU(z);                     
        case 'LReLU'
            alpha=0.01;
            zt = nn.a{i - 1} * nn.W{i - 1}';
            z = max(0 ,zt);
            index= find(z==0);
            z(index)= alpha.*zt(index);
            nn.a{i} =z;
        case 'PReLU' % kaiming He
            alpha=0.01; % learned form the data
            zt = nn.a{i - 1} * nn.W{i - 1}';
            z = max(0 ,zt);
            index= find(z==0);
            z(index)= alpha.*zt(index);
            nn.a{i} =z;
        case 'RReLU'
            if(nn.testing)% at testing time，在测试时用
                % RReLU测试时 alpha为均值
                zt = nn.a{i - 1} * nn.W{i - 1}';
                z = max(0 ,zt);
                index= find(z==0);
                %             nn.alpha{i}=rand(size(zt(index)))*(8-3)+3;
                talpha=11/2;
                z(index)= talpha.*zt(index);
                nn.a{i} =z;
            else
                zt = nn.a{i - 1} * nn.W{i - 1}';
                z = max(0 ,zt);
                index= find(z==0);
                %                 nn.alpha{i}=rand(size(zt(index)))*(8-3)+3;
                nn.alpha{i}=rand(1)*(8-3)+3;
                z(index)= nn.alpha{i}.*zt(index);
                nn.a{i} =z;
            end
        case 'Maxout'
            z = nn.a{i - 1} * nn.W{i - 1}';
            for index0= 1: size(z,1)
                [h(index0),ind]=max((z(index0,:)));
                z(index0,:)=0;
                z(index0,ind)=h(index0);
            end
            nn.maxoutmatrix{i}=(z~=0);
            nn.a{i} =z;
        case 'AMaxout'
            z = nn.a{i - 1} * nn.W{i - 1}';
            for index0= 1: size(z,1)
                [h(index0),ind]=max(abs(z(index0,:)));
                z(index0,:)=0;
                z(index0,ind)=h(index0);
            end
            nn.maxoutmatrix{i}=(z~=0);
            nn.a{i} =z;
        case 'SMaxout'
            if(nn.testing)% at testing time，在测试时用
                % RReLU测试时 alpha为均值
                zt = nn.a{i - 1} * nn.W{i - 1}';
                  nn.a{i} =zt;
                %% 先注释掉下一行， 后面得修改！！！
%                 nn.a{i} =zt.*nn.pp{i};
            else
                z = nn.a{i - 1} * nn.W{i - 1}';
                nn.pp{i}=zeros(size(z)); % 生成同维矩阵，里面的值为概率
                %                 maxoutmatrix=zeros(size(z));
                
                aaa=zeros(size(z)); 
                for index0= 1: size(z,1)
%                     nn.pp{i}(index0,:)= abs(z(index0,:))./sum(abs(z(index0,:)));
                    aaa(index0,:)=abs(z(index0,:))./sum(abs(z(index0,:)));
                    
                    % n is a positive integer specifying the number of trials (sample size) for each multinomial outcome.
                    chose = 1; %  这里只要选1个，所以n置为1
                    % 平均
                    %                     maxoutmatrix(index0,:) = mnrnd(n,nn.pp{i}(index0,:));
                    % 直接学习到的概率系数
                    nn.maxoutmatrix{i}(index0,:) = mnrnd(chose,nn.pp{i}(index0,:));
                    
                    bbb(index0,:) =mnrnd(chose,aaa(index0,:));
                end
                % 平均
                %                 nn.maxoutmatrix{i}=nn.maxoutmatrix{i}+maxoutmatrix;
                nn.a{i} =z.*nn.maxoutmatrix{i};
            end
    end
    
    %dropout hidden layers
    if(nn.dropoutFraction > 0)
        if(nn.testing)
            % 测试时这样做.*(1-p)，也就是测试时候将激活度减半
            nn.a{i} = nn.a{i}.*(1 - nn.dropoutFraction);
        else
            % 针对不同激活函数，因为响应的范围不同，应用均值来作为阈值保证一半的响应被遮蔽
            % nn.dropoutFraction=mean(nn.a{i}(:));

            % dropOutMask{i}为一个和a{i}同维的矩阵，将a{i}中大于nn.dropoutFraction（0.5）的置为1,小于的置为0    
            % 这里用[0,1]均匀分布大于0.5， 已经实现了将一半响应被遮蔽的效果，不必再用均值去截断
            nn.dropOutMask{i} = (rand(size(nn.a{i}))>nn.dropoutFraction);
            % 按均匀分布随机将一半的响应置为0
            nn.a{i} = nn.a{i}.*nn.dropOutMask{i};
        end
    end
    
    
    % calculate running exponential activations for use with sparsity
    % 计算Sparsity，nonSparsityPenalty是对没达到SparsityTarget参数的惩罚系数？
    if(nn.nonSparsityPenalty>0)
        nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(nn.a{i}, 1);% P表示第i层的平均激活度
    end
    
end

% Calculate output of NN
z = bsxfun(@plus,nn.a{n - 1} * nn.W{n - 1}',nn.b{n-1}');
switch nn.output
    case 'sigm'
        nn.a{n} = sigm(z);
    case 'linear'
        nn.a{n} = z;
    case 'softmax'
        %numerically stable calc of softmax
        class_normalizer = log_sum_exp_over_cols(z);
         % 减去一个max输入防止溢出（对结果无影响）
        log_class_prob = bsxfun(@minus,z,class_normalizer);
        % 并做指数变换
        nn.a{n} = exp(log_class_prob);
        %%%OLD CODE
        %nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
        %nn.a{n} = exp(bsxfun(@minus, nn.a{n}, max(nn.a{n},[],2)));
        %nn.a{n} = bsxfun(@rdivide, nn.a{n}, sum(nn.a{n}, 2));        
end

%error and loss 输出层的误差
nn.e = y - nn.a{n};

switch nn.output
    case {'sigm', 'linear'}
        nn.L = 1/2 * sum(sum(nn.e .^ 2)) / m; % Mean square error (MSE)
    case 'softmax'
        %nn.L = -sum(sum(y .* log(nn.a{n}))) / m; %OLD CODE
        nn.L = -sum(sum(y.*log_class_prob)) / m; %mean cross entropy （MCE）
end
end
