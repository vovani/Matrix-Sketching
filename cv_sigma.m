function [ best_sigma, err ] = cv_sigma( method, dataset, s, sigmas, cv_rate )
    cv_s = round(s * (1 - cv_rate));
    parfor iter = 1:length(sigmas)
        tmp_dataset = cv_dataset(dataset, cv_rate);
        tmp_dataset.sigma = sigmas(iter);
        [Z, phi] = method(cv_s, tmp_dataset);
        cv_err(iter) = run_prediction(Z, phi, cv_s, tmp_dataset);
    end
    [~, sigma_id] = min(cv_err);
    best_sigma = sigmas(sigma_id);
end

