function [x, x_sample] = rbmdown(rbm, x)
    x = repmat(rbm.b', size(x, 1), 1) + x * rbm.W;
    switch rbm.vis_units
        case 'sigm'
            x = sigm(x);
            x_sample = sample(x);
        case 'linear'
            % no change, just raw input
            x_sample = x;
        case 'NReLU'
            x = ReLU(x + normrnd(0,sigm(x),size(x)));
            x_sample = x;
        otherwise
            error('Invalid unit type');
    end;
end
