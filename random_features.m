function [ Z, phi ] = random_features( s, dataset )
    d = size(dataset.X, 2);
    gen_random_phi = @(w, b) (@(x) sqrt(2 / s) * cos((x * w)' + repmat(b, [1, size(x, 1)])));

    w = mvnrnd(zeros(s, d), (dataset.sigma ^ -2) * eye(d))';
    b = 2 * pi * rand(s, 1);

    phi = gen_random_phi(w, b);
    Z = phi(dataset.X)';
end

