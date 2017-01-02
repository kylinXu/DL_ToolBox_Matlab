function nn = nnbp_gpu(nn)
%NNBP performs backpropagation
% nn = nnbp(nn) returns an neural network structure with updated weights

n = nn.n;           %number of layers
sparsityError = 0;

% calculate backprop for output layer
switch nn.output    % ��������㼤�����������Ĳв�delta{n}.
    case 'sigm'
        % -��y - a(n)��.*f'(z(n)), where  nn.e = y - nn.a{n}.
        d_act = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
        d{n}    = d_act;
    case {'softmax','linear'}
        d{n} = - nn.e;  % �Եģ�softmax��linear ( f'(z(n))=1 ) ����.
end
%backpropagate trough layers �����ڶ��㵽�ڶ���ĵ���
for i = fliplr(2:n-1) 
     % Derivative of the activation function ������ĵ���
    switch nn.activation_function 
        case 'sigm'
            d_act = nn.a{i} .* (1 - nn.a{i}); % f'(z(i))
        case 'ReLU'  % linear rectified units max(0,x)
            d_act =nn.cast(nn.a{i}>0);
        case 'tanh_opt'
             d_act = nn.cast(2.7159 * 2/3) * (gpuArray.ones(1,nn.caststr) - nn.cast(1/(1.7159).^2) * nn.a{i}.^2);
    end
    
    if(nn.nonSparsityPenalty>0)
        % not tested if sparsitypenalty works with w,b notation
        pi = repmat(nn.p{i}, size(nn.a{i}, 1), 1);
        sparsityError = [zeros(size(nn.a{i},1),1) nn.nonSparsityPenalty * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi))];
    end
    
    % Backpropagate first derivatives
    %
    % i+1=n
    %  d{i} ��ʾ��i��Ĳв� delta
    d{i} = (d{i + 1} * nn.W{i} + sparsityError) .* d_act;
    
    if(nn.dropoutFraction>0)
         % �в��ʱҲҪ���ε���Ӧ��Ӧ�������Ĳвע�⣬ƫ�ã�bias��������в����dropout��
         % �ڶ�ÿ����ݶ�Ȩֵ(����)Ҳ������Ӧ��dropout,����delta���в��ʱҲҪ���ε���Ӧ�Ĳв
        d{i} = d{i} .* nn.dropOutMask{i};
    end
    
end
batchsize = size(d{n},1);
for i = 1 : (n - 1)
        % ����ƫ����������W�ģ�delta��W)
        dt = d{i + 1}';
        nn.dW{i} = (dt * nn.a{i}) / batchsize;
        % nn.db{i} = (dt * ones(size(d{i+1},1),1) / batchsize);
        nn.db{i} = sum(dt,2) / batchsize; % faster than the line above
        clear dt
end
% ��
for u = 1:numel(nn.a)
    nn.a{u} = [];
end
end
