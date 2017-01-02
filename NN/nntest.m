function [er, bad] = nntest(nn, x, y)
    labels = nnpredict(nn, x);
    [~, expected] = max(y,[],2);% �ҵ�y�����������ֵ��index
    bad = find(labels ~= expected);    
    er = numel(bad) / size(x, 1);
end
