function test_matthew
% test matthew correlation function

pred_class    = [ones(10,1);0];
true_class    = [ones(10,1);0];
disp('Test1')
[C,~] = confusionmat(pred_class,true_class);
% tp = 0, tn = 0, fp = 0, fn = 5
[TP,TN,FP,FN,MCC] =  matthew_calc(pred_class,true_class);
my_C = [TP, FN;
        FP, TN];
MCC
assert(isequal(C,my_C));


disp('Test2')
pred_class    = [zeros(10,1);1];
true_class    = [zeros(10,1);1];
% tp = 0, tn = 0, fp = 0, fn = 5
[TP,TN,FP,FN,MCC] =  matthew_calc(pred_class,true_class);
[C,~] = confusionmat(pred_class,true_class);
my_C = [TP, FN;
        FP, TN];
assert(isequal(C,my_C));
MCC

disp('Test3')
pred_class    = [1,1,1,0,0,0,0,0,1,1,1];
true_class    = [1,0,0,1,0,0,0,0,0,1,1];
% tp = 0, tn = 0, fp = 0, fn = 5
[TP,TN,FP,FN,MCC] =  matthew_calc(pred_class,true_class);
[C,~] = confusionmat(pred_class,true_class);
my_C = [TP, FN;
        FP, TN];
assert(isequal(C,my_C));
MCC


    function [TP,TN,FP,FN,MCC] =  matthew_calc(pred_class,true_class)
        TP = sum( (pred_class == true_class) .* (true_class == 0) ); %True positive
        TN = sum( (pred_class == true_class) .* (true_class == 1) ); %True negative
        FP = sum( (pred_class ~= true_class) .* (pred_class == 1) ); %False positive
        FN = sum( (pred_class ~= true_class) .* (pred_class == 0) ); %False negative  
        
        mcc_denom = (TP+FP) * (TP+FN) * (TN+FP) * (TN + FN);
        if mcc_denom == 0
        mcc_denom = 1;
        end
        MCC = (TP * TN - FP * FN) ./ sqrt(mcc_denom);
    end

end

