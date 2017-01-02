function nn = nnff(nn, x, y)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

n = nn.n; % n��ʾ����������������
m = size(x, 1); % ѵ��ʱΪmini-batch��С������������ʱΪ����������С
% x = [ones(m,1) x]; % �����Ƕ�����X����1�У�bias��
nn.a{1} = x;       % ������a1��ʾ����x��Ҳ�������������
cast = @nn.cast;
cstr = nn.caststr;

%feedforward pass
for i = 2 : n-1
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
            %             nn.a{i} = max(0 ,z);
        case 'AReLU'
            % ����ֵ����0��
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
            if(nn.testing)% at testing time���ڲ���ʱ��
                % RReLU����ʱ alphaΪ��ֵ
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
            if(nn.testing)% at testing time���ڲ���ʱ��
                % RReLU����ʱ alphaΪ��ֵ
                zt = nn.a{i - 1} * nn.W{i - 1}';
                  nn.a{i} =zt;
                %% ��ע�͵���һ�У� ������޸ģ�����
%                 nn.a{i} =zt.*nn.pp{i};
            else
                z = nn.a{i - 1} * nn.W{i - 1}';
                nn.pp{i}=zeros(size(z)); % ����ͬά���������ֵΪ����
                %                 maxoutmatrix=zeros(size(z));
                
                aaa=zeros(size(z)); 
                for index0= 1: size(z,1)
%                     nn.pp{i}(index0,:)= abs(z(index0,:))./sum(abs(z(index0,:)));
                    aaa(index0,:)=abs(z(index0,:))./sum(abs(z(index0,:)));
                    
                    % n is a positive integer specifying the number of trials (sample size) for each multinomial outcome.
                    chose = 1; %  ����ֻҪѡ1��������n��Ϊ1
                    % ƽ��
                    %                     maxoutmatrix(index0,:) = mnrnd(n,nn.pp{i}(index0,:));
                    % ֱ��ѧϰ���ĸ���ϵ��
                    nn.maxoutmatrix{i}(index0,:) = mnrnd(chose,nn.pp{i}(index0,:));
                    
                    bbb(index0,:) =mnrnd(chose,aaa(index0,:));
                end
                % ƽ��
                %                 nn.maxoutmatrix{i}=nn.maxoutmatrix{i}+maxoutmatrix;
                nn.a{i} =z.*nn.maxoutmatrix{i};
            end
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
            nn.dropOutMask{i} = (rand(size(nn.a{i}))>nn.dropoutFraction);
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
        %numerically stable calc of softmax
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
