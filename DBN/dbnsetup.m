function dbn = dbnsetup(dbn, x, opts)

    rand_weight_sigma = 0.01;

    n = size(x, 2);
    dbn.sizes = [n, dbn.sizes];
    
    %defaults
    %% 默认的激活函数
    if (~isfield(opts,'vis_units') || isempty(opts.vis_units))
        opts.vis_units = 'sigm'; 
    end;
    if (~isfield(opts,'hid_units') || isempty(opts.hid_units))
        opts.hid_units = 'sigm';
    end;
    %% 动量
    if (~isfield(opts,'momentum') || isempty(opts.momentum))
        opts.momentum = 0;
    end;
    %% cdk
    if (~isfield(opts,'cdn') || isempty(opts.momentum))
        opts.cdn = 1;
    end;
    
%      直接分层初始化每一层的rbm(受限波尔兹曼机(Restricted Boltzmann Machines, RBM))
%      同样，W,b,c是参数，vW,vb,vc是更新时用到的与momentum的变量，见到代码时再说
    for u = 1 : numel(dbn.sizes) - 1
        dbn.rbm{u}.alpha    = opts.alpha;
        dbn.rbm{u}.momentum = opts.momentum;
        dbn.rbm{u}.cdn      = opts.cdn;% CDk 算法 k=1
        
        % make vis_units only actually visible units (1st layer) 激活函数选择
        if (u == 1)
            dbn.rbm{u}.vis_units = opts.vis_units;
        else
            dbn.rbm{u}.vis_units = opts.hid_units;
        end;
        dbn.rbm{u}.hid_units = opts.hid_units;

        % weights 权重
        dbn.rbm{u}.W  = normrnd(0, rand_weight_sigma, dbn.sizes(u + 1), dbn.sizes(u)); % normal distribution.
        dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));% 动量更新时用到的变量

        % visible biases 输入层偏置
        dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1);
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);

        % hidden biases  隐层偏置
        dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1);
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
    end

end
