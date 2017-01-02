function dbn = dbnsetup(dbn, x, opts)

    rand_weight_sigma = 0.01;

    n = size(x, 2);
    dbn.sizes = [n, dbn.sizes];
    
    %defaults
    %% Ĭ�ϵļ����
    if (~isfield(opts,'vis_units') || isempty(opts.vis_units))
        opts.vis_units = 'sigm'; 
    end;
    if (~isfield(opts,'hid_units') || isempty(opts.hid_units))
        opts.hid_units = 'sigm';
    end;
    %% ����
    if (~isfield(opts,'momentum') || isempty(opts.momentum))
        opts.momentum = 0;
    end;
    %% cdk
    if (~isfield(opts,'cdn') || isempty(opts.momentum))
        opts.cdn = 1;
    end;
    
%      ֱ�ӷֲ��ʼ��ÿһ���rbm(���޲���������(Restricted Boltzmann Machines, RBM))
%      ͬ����W,b,c�ǲ�����vW,vb,vc�Ǹ���ʱ�õ�����momentum�ı�������������ʱ��˵
    for u = 1 : numel(dbn.sizes) - 1
        dbn.rbm{u}.alpha    = opts.alpha;
        dbn.rbm{u}.momentum = opts.momentum;
        dbn.rbm{u}.cdn      = opts.cdn;% CDk �㷨 k=1
        
        % make vis_units only actually visible units (1st layer) �����ѡ��
        if (u == 1)
            dbn.rbm{u}.vis_units = opts.vis_units;
        else
            dbn.rbm{u}.vis_units = opts.hid_units;
        end;
        dbn.rbm{u}.hid_units = opts.hid_units;

        % weights Ȩ��
        dbn.rbm{u}.W  = normrnd(0, rand_weight_sigma, dbn.sizes(u + 1), dbn.sizes(u)); % normal distribution.
        dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));% ��������ʱ�õ��ı���

        % visible biases �����ƫ��
        dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1);
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);

        % hidden biases  ����ƫ��
        dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1);
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
    end

end
