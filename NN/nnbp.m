function nn = nnbp(nn)
%NNBP performs backpropagation
% nn = nnbp(nn) returns an neural network structure with updated weights

n = nn.n;           %number of layers
sparsityError = 0;
switch nn.output    % 根据输出层激活函数求出输出层的残差delta{n}
    case 'sigm'
        % -（y - a(n)）.*f'(z(n)), where  nn.e = y - nn.a{n}.
        d_act = - nn.e .* (nn.a{n} .* (1 - nn.a{n})); 
        d{n}    = d_act;
    case {'softmax','linear'}
        d{n} = - nn.e; % 对的，softmax和linear类似
end
%backpropagate trough layers 倒数第二层到第二层的导数
for i = fliplr(2:n-1)
    % Derivative of the activation function 激活函数的导数
    switch nn.activation_function
        case 'sigm'
            d_act = nn.a{i} .* (1 - nn.a{i}); % f'(z(i))
        case 'ReLU'  % linear rectified units max(0,x)
            d_act = nn.cast(nn.a{i}>0);
            %             d_act = nn.a{i};
            %             index= find (nn.a{i}>0);
            %             d_act(index)=1;
        case 'AReLU'
            d_act = nn.a{i};
            % 绝对值大于0的
            index= find (abs(nn.a{i})>0);
            d_act(index)=1;
        case 'RReLU'
            d_act = nn.a{i};
            index= find (nn.a{i}(:,2:end)>0);
            d_act(index)=1;
            index= find (nn.a{i}(:,2:end)<0);
            d_act(index)=nn.alpha{i};
        case 'Maxout'
            % max（x）函数对x求导取最大置为1
            z=nn.a{i};
            for index0 = 1: size(z,1)
                [h(index0),ind]=max((z(index0,:)));
                z(index0,:)=0;
                z(index0,ind)=1;% 置为1
            end
            d_act =z;
        case 'AMaxout'
            % max（x）函数对x求导取最大置为1
            z=nn.a{i};
            for index0 = 1: size(z,1)
                [h(index0),ind]=max(abs(z(index0,:)));
                z(index0,:)=0;
                z(index0,ind)=1;% 置为1
            end
            d_act =z;
            %               d_act=[ones(size(nn.a{i},1),1) nn.maxoutmatrix{i}];
        case 'SMaxout'
            % max（x）函数对x求导取最大置为1
            z=nn.a{i};
            for index0 = 1: size(z,1)
                [h(index0),ind]=max(abs(z(index0,:)));
                z(index0,:)=0;
                z(index0,ind)=1;% 置为1
            end
            d_act =z;
        case 'tanh_opt'
            d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i}.^2);
    end
    
    if(nn.nonSparsityPenalty>0)
        % not tested if sparsitypenalty works with w,b notation
        pi = repmat(nn.p{i}, size(nn.a{i}, 1), 1);
        sparsityError = [zeros(size(nn.a{i},1),1) nn.nonSparsityPenalty * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi))];
    end
  
    %% old code
%     % Backpropagate first derivatives
%     % d{i} 表示第i层的残差 delta
%     % 输出层
%     if i+1==n % in this case in d{n} there is not the bias term to be removed，最后一层没有偏置
%         d{i} = (d{i + 1} * nn.W{i} + sparsityError) .* d_act; % Bishop (5.56)%delta(L),also called residual
%         % 其他层
%     else % in this case in d{i} the bias term has to be removed，偏置不参与误差反传
%         d{i} = (d{i + 1}(:,2:end) * nn.W{i} + sparsityError) .* d_act; % 乘d_act时候考虑了bias
%     end
%     
%     if(nn.dropoutFraction>0) % i从2开始，ones(size(d{i},1),1)=100*1
%         % 残差反传时也要屏蔽掉相应响应所带来的残差。注意，偏置（bias）不参与残差反传及dropout。
%         d{i} = d{i} .* [ones(size(d{i},1),1) nn.dropOutMask{i}];%在对每层的梯度权值(矩阵)也得做相应的dropout,计算delta（残差反传时也要屏蔽掉相应的残差）
%     end
    
    % Backpropagate first derivatives
    %  d{i} 表示第i层的残差 delta
    d{i} = (d{i + 1} * nn.W{i} + sparsityError) .* d_act; % Bishop (5.56)
        
    if(nn.dropoutFraction>0) % i从2开始，ones(size(d{i},1),1)=100*1
         % 残差反传时也要屏蔽掉相应响应所带来的残差。注意，偏置（bias）不参与残差反传及dropout。
         % 在对每层的梯度权值(矩阵)也得做相应的dropout,计算delta（残差反传时也要屏蔽掉相应的残差）
        d{i} = d{i} .* nn.dropOutMask{i};
    end
    
end

batchsize = size(d{n},1);
for i = 1 : (n - 1)
    % 计算偏导数（关于W的）delta（W)
    dt = d{i + 1}';
    nn.dW{i} = (dt * nn.a{i}) / batchsize;
    % nn.db{i} = (dt * ones(size(d{i+1},1),1) / batchsize);
    nn.db{i} = sum(dt,2) / batchsize; % faster than the line above
end
end