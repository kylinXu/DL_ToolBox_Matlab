function nnplottest(nn,fhandle,L,opts,i)
%NNPLOTTEST plots training and missclassification rate
% Plots all coefficients and training error. Used with opts.errfun set to
% @matthew.

%

%    plotting
figure(fhandle);

x_ax = 1:i;     %create axis

if opts.validation == 1
    
    % tranining error plot
    subplot(1,2,1);
    p = semilogy(x_ax, L.train.e(x_ax), 'b', ...
        x_ax, L.val.e(x_ax), 'r');
    grid on;
    legend(p, {'Training', 'Validation'},'Location','NorthEast');
    switch nn.output
        case {'sigm', 'linear'}
            xlabel('Number of epochs'); ylabel('MSE');title('Training Error');
        case 'softmax'
            xlabel('Number of epochs'); ylabel('MCE');title('Training Error');
    end
    set(gca, 'Xlim',[0,opts.numepochs + 1])
    %create subplots of correlations
    
    subplot(1,2,2);
    p = semilogy(x_ax, L.train.e_errfun(x_ax,1), 'b', ...
        x_ax, L.val.e_errfun(x_ax,1),   'm');
     grid on;
    
    title('Missclassification rate')
    ylabel('Missclassification'); xlabel('Number of epochs');
    legend(p, {'Training', 'Validation'},'Location','NorthEast');
    set(gca, 'Xlim',[0,opts.numepochs + 1])
      
else  % no validation
    subplot(1,2,1);
    title('Training Errors')
    p = semilogy(x_ax,L.train.e(x_ax),'b');
    legend(p, {'Training'},'Location','NorthEast');
     grid on;
    switch nn.output
        case {'sigm', 'linear'}
            xlabel('Number of epochs'); ylabel('MSE');title('Training Error');
        case 'softmax'
            xlabel('Number of epochs'); ylabel('MCE');title('Training Error');
    end
    set(gca, 'Xlim',[0,opts.numepochs + 1])
     
    subplot(1,2,2);
    p = semilogy(x_ax, L.train.e_errfun(x_ax,1), 'b');
     grid on;
    ylabel('Misclassification'); xlabel('Number of epochs');
    title('Misclassification rate')
    legend(p, {'Training'},'Location','NorthEast');
    set(gca, 'Xlim',[0,opts.numepochs + 1])
    
end

drawnow;

end