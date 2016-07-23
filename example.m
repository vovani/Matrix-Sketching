all_datasets = {load('acoustic'), load('adult'), load('cadata'), ...
                load('cod-rna'), load('cpu'), load('ijcnn1'), load('mnist')};
            
cv_rate = 0.9;
for i = 2:length(all_datasets)
    i
    dataset = all_datasets{i};
    s = 500;
    sigmas = round(2 .^ [0 : 0.5 : 7]);
    
    fprintf('Running RF\n');
    random_sigma(i) = cv_sigma(@random_features, dataset, s, sigmas, cv_rate);
    dataset.sigma = random_sigma(i);
    [Z, phi] = random_features(s, dataset);
    random_err(i) = run_prediction(Z, phi, s, dataset);
    [random_sigma(i), random_err(i)]
    
    fprintf('Running Nystrom\n');
    nystrom_sigma(i) = cv_sigma(@nystrom, dataset, s, sigmas, cv_rate);
    dataset.sigma = nystrom_sigma(i);
    [Z, phi] = nystrom(s, dataset);
    nystrom_err(i) = run_prediction(Z, phi, s, dataset);
    [nystrom_sigma(i), nystrom_err(i)]
end
