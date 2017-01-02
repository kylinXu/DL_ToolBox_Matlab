function nn = nnff_gpu(nn, x, y)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

n = nn.n;             % n��ʾ����������������
m = size(x, 1);       % ѵ��ʱΪmini-batch��С������������ʱΪ����������С
% x = [ones(m,1) x];  % �����Ƕ�����X����1�У�bias��
nn.a{1} = x;          % ������a1��ʾ����x��Ҳ�������������
cast = @nn.cast;
cstr = nn.caststr;

%% feedforward pass
for i = 2 : n-1
    
    %% calculate activation of each layer, except output layer
    % ����ʵ�ֵļǷ���X��m*nά��mΪ����������nΪ����ά���� W��k*nά��kΪ����ڵ�����
    z = bsxfun(@plus, nn.a{i - 1} * nn.W{i - 1}',nn.b{i-1}'); %input to each layer
    switch nn.activation_function
        case 'sigm'
            % Calculate the unit's outputs (including the bias term)
            nn.a{i} = sigm(z);
        case 'tanh_opt'
            nn.a{i} = tanh_opt(z);
        case 'ReLU'  % linear rectified units max(0,x)
            nn.a{i} = ReLU(z);
    end
    
    %dropout hidden layers
    if(nn.dropoutFraction > 0)
        if(nn.testing)
            % ����ʱ������.*(1-p)��Ҳ���ǲ���ʱ�򽫼���ȼ���
            nn.a{i} = nn.a{i}.*(1 - nn.dropoutFraction);
        else
            % ��Բ�ͬ���������Ϊ��Ӧ�ķ�Χ��ͬ��Ӧ�þ�ֵ����Ϊ��ֵ��֤һ�����Ӧ���ڱ�
            % nn.dropoutFraction=mean(nn.a{i}(:));

            % dropOutMask{i}Ϊһ����a{i}ͬά�ľ��󣬽�a{i}�д���nn.dropoutFraction��0.5������Ϊ1,С�ڵ���Ϊ0    
            % ������[0,1]���ȷֲ�����0.5�� �Ѿ�ʵ���˽�һ����Ӧ���ڱε�Ч�����������þ�ֵȥ�ض�
            nn.dropOutMask{i} = gpuArray.rand(size(nn.a{i}),cstr)>nn.dropoutFraction;
            % �����ȷֲ������һ�����Ӧ��Ϊ0
            nn.a{i} = nn.a{i}.*nn.dropOutMask{i};
        end
    end
    
    
    % calculate running exponential activations for use with sparsity
    % ����Sparsity��nonSparsityPenalty�Ƕ�û�ﵽSparsityTarget�����ĳͷ�ϵ����
    if(nn.nonSparsityPenalty>0)
        nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(nn.a{i}, 1);% P��ʾ��i���ƽ�������
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
        % numerically stable calc of softmax
        class_normalizer = log_sum_exp_over_cols(z);
        % ��ȥһ��max�����ֹ������Խ����Ӱ�죩
        log_class_prob = bsxfun(@minus,z,class_normalizer);
        % ����ָ���任
        nn.a{n} = exp(log_class_prob);
        %%%OLD CODE
        %nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
        %nn.a{n} = exp(bsxfun(@minus, nn.a{n}, max(nn.a{n},[],2)));
        %nn.a{n} = bsxfun(@rdivide, nn.a{n}, sum(nn.a{n}, 2));
        
end

%error and loss ���������
nn.e = y - nn.a{n};

switch nn.output
    case {'sigm', 'linear'}
        nn.L = 1/2 * sum(sum(nn.e .^ 2)) / m; % Mean square error (MSE)
    case 'softmax'
        %nn.L = -sum(sum(y .* log(nn.a{n}))) / m; %OLD CODE
        nn.L = -sum(sum(y.*log_class_prob)) / m; %mean cross entropy ��MCE��
end
end
