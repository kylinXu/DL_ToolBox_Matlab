function dbn = dbntrain(dbn, x, opts)
    n = numel(dbn.rbm);% 网络层数（不算输入层）
    % 对每一层的rbm进行训练  
    dbn.rbm{1} = rbmtrain(dbn.rbm{1}, x, opts);% 训练一个RBM
    for i = 2 : n % 每一层
        x = rbmup(dbn.rbm{i - 1}, x);% 将下一层输出作为输入
        dbn.rbm{i} = rbmtrain(dbn.rbm{i}, x, opts); 
    end

end
