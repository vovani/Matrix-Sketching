function [Z, phi] = nystrom(s, dataset)
    gaussian_kernel_gram = @(x, z) exp( - (pdist2(x, z, 'euclidean') .^ 2) ./ (2 * (dataset.sigma ^ 2)));

    indecies = randsample(1 : size(dataset.X, 1), s);
    s_phi = @(z) gaussian_kernel_gram(dataset.X(indecies,:), z);
    subsampled_kernel = s_phi(dataset.X(indecies,:));
    R = chol(subsampled_kernel + 0.0001 * eye(s));

    Rt_inv = inv(R');
    phi = @(x) Rt_inv * s_phi(x);
    Z = (Rt_inv *  s_phi(dataset.X))';
end
