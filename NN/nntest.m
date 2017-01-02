function [er, bad] = nntest(nn, x, y)
    labels = nnpredict(nn, x);
    [~, expected] = max(y,[],2);% 找到y行向量中最大值得index
    bad = find(labels ~= expected);    
    er = numel(bad) / size(x, 1);
end
