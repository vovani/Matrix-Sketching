function err = run_prediction( Z, phi, s, dataset )
    if isfield(dataset, 'Y')
        f = (Z' * Z + dataset.lambda * eye(s)) \ (Z' * dataset.Y);
        reg = phi(dataset.Xt)' * f;
        err = norm(reg - dataset.Yt, 2) / norm( dataset.Yt, 2 );
    else
        classes = unique(dataset.L)';
        f = zeros(length(classes), s);
        for i = 1:length(classes)
            c = classes(i);
            one2rest = 2 * (dataset.L == c) - 1;
            f(i,:) = (Z' * Z + dataset.lambda * eye(s)) \ (Z' * one2rest);
        end 

        [~, pred] =  max(f * phi(dataset.Xt));
        err = nnz(classes(pred) ~= dataset.Lt') / length(dataset.Lt);
    end 
end

