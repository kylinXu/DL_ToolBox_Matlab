function nn = dbnunfoldtonn(dbn, outputsize)
%DBNUNFOLDTONN Unfolds a DBN to a NN
%   dbnunfoldtonn(dbn, outputsize ) returns the unfolded dbn with a final
%   layer of size outputsize added.
%   outputsize是你的目标输出label，比如在MINST就是10，DBN只负责学习feature  
%   或者说初始化Weight，是一个unsupervised learning，最后的supervised还得靠NN  
    if(exist('outputsize','var'))
        size = [dbn.sizes outputsize];
    else
        size = [dbn.sizes];
    end
    nn = nnsetup(size);
    for i = 1 : numel(dbn.rbm)
        nn.W{i} = dbn.rbm{i}.W;
        nn.b{i} = dbn.rbm{i}.c;
    end
end

