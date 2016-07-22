all_datasets = {load('acoustic'), load('adult'), load('cadata'), ...
                load('cod-rna'), load('cpu'), load('ijcnn1'), load('mnist')};
for i = 2:length(all_datasets)
    i
    dataset = all_datasets{i};
    n = size(dataset.X, 1);
    s = 1000;
    
    fprintf('Running RF');
    random_iter = 0;
    random_sigmas =  2 .^ (1 : 6);
    for sigma = random_sigmas;
        random_iter = random_iter + 1;
        dataset.sigma = sigma;
        [random_Z, random_phi] = random_features(s, dataset);
        random_err(i,random_iter) = run_prediction(random_Z, random_phi, s, dataset);
    end

    fprintf('Running Nystrom');
    nystrom_iter = 0;
    nystrom_sigmas = 2 .^ (1 : 6);
    for sigma = nystrom_sigmas
        nystrom_iter = nystrom_iter + 1;
        dataset.sigma = sigma;
        [nystrom_Z, nystrom_phi] = nystrom(s, dataset);
        nystrom_err(i, nystrom_iter) = run_prediction(nystrom_Z, nystrom_phi, s, dataset);
    end
end
