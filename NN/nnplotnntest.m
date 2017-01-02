function nnplotnntest(nn,fhandle,L,opts,i)
%NNPLOTNNTEST updates figures during training
% plots the missclassification rate, to be used with nntest as error function    

    %    plotting
    figure(fhandle); 
    subplot(1,2,1)
    x_ax = 1:i;
    if opts.validation == 1 
        p = plot(x_ax, L.train.e(x_ax), 'b', ...
                 x_ax, L.val.e(x_ax), 'r');
        legend(p, {'Training', 'Validation'},'Location','NorthEast');
    else
        p = plot(x_ax,L.train.e(x_ax),'b');
        legend(p, {'Training'},'Location','NorthEast');
    end    
    xlabel('Number of epochs'); ylabel('Error');title('Error');    
    set(gca, 'Xlim',[0,opts.numepochs + 1])

    if i ==1 % speeds up plotting by factor of ~2
        set(gca,'LegendColorbarListeners',[]);
        setappdata(gca,'LegendColorbarManualSpace',1);
        setappdata(gca,'LegendColorbarReclaimSpace',1);

    end
    
    
    
    subplot(1,2,2)
    if opts.validation == 1 
        p = plot(x_ax, L.train.e_errfun(x_ax,:), 'b', ...
                 x_ax, L.val.e_errfun(x_ax,:), 'r');
        legend(p, {'Training', 'Validation'},'Location','NorthEast');
    else
        p = plot(x_ax,L.train.e_errfun(x_ax,:),'b');
        legend(p, {'Training'},'Location','NorthEast');
    end    
    xlabel('Number of epochs'); ylabel('Misclassification %');title('Misclassification rate');    
    set(gca, 'Xlim',[0,opts.numepochs + 1])

    if i ==1 % speeds up plotting by factor of ~2
        set(gca,'LegendColorbarListeners',[]);
        setappdata(gca,'LegendColorbarManualSpace',1);
        setappdata(gca,'LegendColorbarReclaimSpace',1);

    end
    
    drawnow;
end