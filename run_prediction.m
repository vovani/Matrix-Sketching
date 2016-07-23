function err = run_prediction( Z, phi, s, dataset )
    if isfield(dataset, 'Y')
        f = (Z' * Z + dataset.lambda * eye(s)) \ (Z' * dataset.Y);
        reg = phi(dataset.Xt)' * f;
        err = norm(reg - dataset.Yt, 2) / norm( dataset.Yt, 2 );
    else
        classes = unique(dataset.L)';
        iter = 0;
        for i = 1:length(classes)
            for j = i + 1: length(classes)
                ci = classes(i);
                cj = classes(j);
                Zi = Z(dataset.L == ci, :);
                Zj = Z(dataset.L == cj, :);
                A = Zi' * Zi + Zj' * Zj;
                y = Zi' * ones(size(Zi,1), 1) - Zj' * ones(size(Zj,1), 1);
                
                iter = iter + 1;
                in(iter, :) = [ci,cj];
                f(iter,:) = (A + dataset.lambda * eye(s)) \ y;
            end
        end 

        pos = repmat(in(:, 1), [1 size(dataset.Xt,1)]);
        neg = repmat(in(:, 2), [1 size(dataset.Xt,1)]);
        
        t = f * phi(dataset.Xt);
        
        pred = neg .* (t < 0) + pos.* (t>0);
        if size(t,1) > 1
            pred = mode(pred);
        end
        err = nnz(pred ~= dataset.Lt') / length(dataset.Lt);
    end 
end

