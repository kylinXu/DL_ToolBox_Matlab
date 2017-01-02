function sae = saetrain_gpu(sae, x, opts)
    for i = 1 : numel(sae.ae);
        disp(['Training AE ' num2str(i) '/' num2str(numel(sae.ae))]);
        sae.ae{i} = nntrain_gpu(sae.ae{i}, x, x, opts);
        t = nnff(sae.ae{i}, x, x);
        x = t.a{2};
        %remove bias term !!! NOT NESCESSARY DUE TO W+b notation
        %x = x(:,2:end);
    end
end
