import os

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch

from mining import estimate, datasets
from mining.estimate import prep_gp_data, train_gaussian_process, get_gp_means, get_gp_models, get_splines, get_spline_means

def get_expected_class(model_output):
    # Convert to probabilities using exp
    class_probs = torch.softmax(model_output, dim=-1)
    # Calculate expected class
    expected_class = torch.sum(class_probs * torch.arange(class_probs.shape[-1]), dim=-1)
    
    expected_mean = expected_class - 1
    return expected_mean

def plot_gp(
    model,
    likelihood,
    train_x=None,
    train_y=None,
    plot_observed_data=False,
    plot_predictions=False,
    n_prior_samples=0,
    n_test=10,
    ax=None,
):
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        if plot_observed_data:
            # Plot training data as black stars
            ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        if plot_predictions:
            min_x = torch.min(train_x)
            max_x = torch.max(train_x)
            start_x = min_x - 0.1 * (max_x - min_x)
            end_x = max_x + 0.1 * (max_x - min_x)
            test_x = torch.linspace(start_x, end_x, n_test).to('cuda')
            test_classifier_ids = torch.zeros(n_test, dtype=torch.long)
            model_pred = model(test_x)
            # likelihood.set_classifier_ids(test_classifier_ids)
            # observed_pred = likelihood(model_pred)
            # Get upper and lower confidence bounds
            lower, upper = model_pred.confidence_region()
            # Plot predictive means as blue line
            # mean = get_expected_class(model_pred.loc.T)
            ax.plot(test_x.cpu().numpy(), model_pred.loc.cpu().numpy(), 'b')
            # lower = get_expected_class(lower.T)
            # upper = get_expected_class(upper.T)
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])

def plot_spline(
    timeline, mean, confidence_interval,
    train_x=None,
    train_y=None,
    plot_observed_data=False,
    plot_predictions=False,
    n_test=100,
    ax=None,
):

    # Initialize plot
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

    if plot_observed_data:
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    if plot_predictions:
        # min_x = torch.min(train_x)
        # max_x = torch.max(train_x)
        # start_x = min_x - 0.1 * (max_x - min_x)
        # end_x = max_x + 0.1 * (max_x - min_x)
        # test_x = torch.linspace(start_x, end_x, n_test)
        # test_y = model.predict(test_x.reshape(-1, 1)).reshape(-1)  
        # Plot predictive means as blue line
        # mean = get_expected_class(model_pred.loc.T)
        # ax.plot(test_x.numpy(), test_y, 'b')
        ax.plot(timeline, mean, 'b')
        ax.fill_between(timeline, confidence_interval[:,0], confidence_interval[:,1], alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])

def main():
    num_data_points = 10
    user_stance = np.linspace(-1, 1, num_data_points)
    user_stance_variance = 0.1
    predict_profile = 'low_recall'
    dataset = datasets.SimpleGenerativeOpinionTimelineDataset(user_stance, user_stance_variance, num_data_points, pred_profile_type=predict_profile, halflife=10.)

    X_norm, X, y, classifier_ids = prep_gp_data(dataset)
    all_classifier_profiles = dataset.all_classifier_profiles

    time_span_ratio = (torch.max(X) - torch.min(X)) / (torch.max(X_norm) - torch.min(X_norm))
    lengthscale_loc = time_span_ratio * 0.01
    lengthscale_scale = time_span_ratio * 0.1

    model_type = 'gp'
    if model_type == 'gp':
        model_list, likelihood_list, model_map, train_xs, train_ys = get_gp_models(
            X_norm, 
            y, 
            classifier_ids, 
            all_classifier_profiles, 
            lengthscale_loc=lengthscale_loc, 
            lengthscale_scale=lengthscale_scale,
            gp_type='ordinal'
        )

        # get some samples from the untrained models
        # fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        # for idx in range(min(3, len(model_list.models))):
        #     i, k = model_map[idx]
        #     model = model_list.models[idx]
        #     likelihood = likelihood_list.likelihoods[idx]
        #     plot(model, likelihood, train_x=X_norm[i,:], train_y=y[i,:,k], plot_observed_data=True, plot_predictions=True, ax=ax)
        # plt.show()
        
        models, likelihood_list, model_map, losses = train_gaussian_process(
            X_norm, 
            y, 
            classifier_ids, 
            all_classifier_profiles, 
            lengthscale_loc=lengthscale_loc, 
            lengthscale_scale=lengthscale_scale,
            gp_type='ordinal',
        )
    elif model_type == 'spline':
        models, coef_bootstraps, spline_transformers, model_map = get_splines(X_norm, y, alpha=1e-1)
        timelines, means, confidence_intervals = get_spline_means(dataset, models, coef_bootstraps, spline_transformers, model_map, X_norm, X, y)

    # get some samples from the untrained models
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    for idx in range(min(3, len(models))):
        if model_type == 'gp':
            i, k = model_map[idx]
            model = model_list[idx]
            likelihood = likelihood_list[idx]
            plot_gp(model, likelihood, train_x=X_norm[i,:], train_y=y[i,:,k], plot_observed_data=True, plot_predictions=True, ax=ax)
        elif model_type == 'spline':
            i, k = model_map[idx]
            # model = model_list[idx]
            timeline, mean, confidence_interval = timelines[i], means[i,:,k], confidence_intervals[i,:,k]
            plot_spline(timeline, mean, confidence_interval, train_x=X[i,:], train_y=y[i,:,k], plot_observed_data=True, plot_predictions=True, ax=ax)
    plt.show()

    # timestamps, means = get_gp_means(dataset, model_list, likelihood_list, model_map, X_norm, X, y)



if __name__ == "__main__":
    main()