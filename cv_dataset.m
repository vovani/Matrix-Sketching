function [ cv_dataset ] = cv_dataset( orig_dataset, cv_rate )
    cv_dataset = orig_dataset;
    cv_indecies = crossvalind('HoldOut', size(orig_dataset.X, 1), cv_rate);
    cv_dataset.X = orig_dataset.X(cv_indecies, :);
    cv_dataset.Xt = orig_dataset.X(~cv_indecies, :);
    if isfield(orig_dataset, 'Y')
        cv_dataset.Y = orig_dataset.Y(cv_indecies, :);
        cv_dataset.Yt = orig_dataset.Y(~cv_indecies, :);
    else
        cv_dataset.L = orig_dataset.L(cv_indecies, :);
        cv_dataset.Lt = orig_dataset.L(~cv_indecies, :);
    end
end

