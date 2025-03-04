import datetime
import os

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from mining.datasets import RedditOpinionTimelineDataset
from mining.dynamics import flow_pairplot
from mining.estimate import prep_gp_data, train_gaussian_process, get_gp_means, get_splines, get_spline_means

def plot_spline(
    timeline, mean, confidence_interval,
    train_x=None,
    train_y=None,
    plot_observed_data=False,
    plot_predictions=False,
    plot_legend=False,
    n_test=100,
    ax=None,
    opinion_name=None
):

    # Initialize plot
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

    if plot_observed_data:
        # Plot training data as black stars
        ax.plot([datetime.datetime.fromtimestamp(int(t)) for t in train_x.numpy()], train_y.numpy(), 'k*')
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
        ax.plot([datetime.datetime.fromtimestamp(int(t)) for t in timeline], mean, 'b')
        ax.fill_between([datetime.datetime.fromtimestamp(int(t)) for t in timeline], confidence_interval[:,0], confidence_interval[:,1], alpha=0.5)
    ax.set_ylim([-1.5, 1.5])
    if plot_legend:
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
    
    ax.set_title(opinion_name)

def plot(
    model,
    likelihood,
    train_x=None,
    train_y=None,
    plot_observed_data=False,
    plot_predictions=False,
    plot_legend=False,
    n_prior_samples=0,
    n_test=500,
    ax=None,
    opinion_name=None
):
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Initialize plot
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(4, 3))

        if plot_observed_data:
            # Plot training data as black stars
            ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        if plot_predictions:
            min_x = torch.min(train_x)
            max_x = torch.max(train_x)
            start_x = min_x - 0.1 * (max_x - min_x)
            end_x = max_x + 0.1 * (max_x - min_x)
            test_x = torch.linspace(start_x, end_x, n_test)
            if torch.cuda.is_available():
                test_x = test_x.cuda()
            test_classifier_ids = torch.zeros(n_test, dtype=torch.long)
            batch_size = 10
            batches = [test_x[i:i+batch_size] for i in range(0, len(test_x), batch_size)]
            with gpytorch.settings.cholesky_max_tries(5):
                model_pred = model(batches[0])
                loc = model_pred.loc
                lower, upper = model_pred.confidence_region()
                for batch in batches[1:]:
                    model_pred = model(batch)
                    loc = torch.cat([loc, model_pred.loc], dim=0)
                    # Get upper and lower confidence bounds
                    batch_lower, batch_upper = model_pred.confidence_region()
                    lower = torch.cat([lower, batch_lower], dim=0)
                    upper = torch.cat([upper, batch_upper], dim=0)
            # Plot predictive means as blue line
            if torch.cuda.is_available():
                test_x = test_x.cpu()
                loc = loc.cpu()
                lower = lower.cpu()
                upper = upper.cpu()
            ax.plot(test_x.numpy(), loc.numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        if plot_legend:
            ax.legend(['Observed Data', 'Mean', 'Confidence'])
        ax.set_title(opinion_name)

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir_path = os.path.join(this_dir_path, "..", "..")

    data_source = "reddit"
    source_dir = "4sub_1year"
    if source_dir == "4sub_1year":
        topics_dir_path = os.path.join(root_dir_path, "data", data_source, source_dir)
    elif source_dir == "1sub_1year":
        topics_dir_path = os.path.join(root_dir_path, "data", data_source, source_dir, "topics_minilm_0_2")
    subsample_users = None
    dataset = RedditOpinionTimelineDataset(topics_dir_path, subsample_users=subsample_users, seed=0)

    all_classifier_profiles = dataset.all_classifier_profiles
    stance_targets = dataset.stance_columns
    stance_names = ['Vaccine Mandates', 'Renter Protections', 'NDP', 'Liberals', 'Conservatives', 'Gun Control', 'Drug Decriminalization', 'Liberal Immigration Policy', 'Canadian Aid to Ukraine']
    X_norm, X, y, classifier_ids = prep_gp_data(dataset)

    model_type = 'spline'
    if model_type == 'gp':
        max_x = torch.max(torch.nan_to_num(X, 0))
        min_x = torch.min(torch.nan_to_num(X, torch.inf))
        max_x_norm = torch.max(torch.nan_to_num(X_norm, 0))
        min_x_norm = torch.min(torch.nan_to_num(X_norm, torch.inf))
        time_span_ratio = (max_x - min_x) / (max_x_norm - min_x_norm)
        month_span_ratio = time_span_ratio / (60 * 60 * 24 * 30)
        lengthscale_loc = month_span_ratio * 0.5
        lengthscale_scale = month_span_ratio * 0.1

        models, likelihood_list, model_map, losses = train_gaussian_process(X_norm, y, classifier_ids, stance_targets, all_classifier_profiles, lengthscale_loc=lengthscale_loc, lengthscale_scale=lengthscale_scale, gp_type='ordinal')
        timestamps, means, confidence_region = get_gp_means(dataset, models, likelihood_list, model_map, X_norm, X, y)
    elif model_type == 'spline':
        models, coef_bootstraps, spline_transformers, model_map = get_splines(X_norm, y, alpha=2e-1, n_knots=4)
        timestamps, means, confidence_region = get_spline_means(dataset, models, coef_bootstraps, spline_transformers, model_map, X_norm, X, y)

    os.makedirs(os.path.join(root_dir_path, "data", data_source, source_dir, model_type), exist_ok=True)
    np.save(os.path.join(root_dir_path, "data", data_source, source_dir, model_type, "timestamps.npy"), timestamps)
    np.save(os.path.join(root_dir_path, "data", data_source, source_dir, model_type, "means.npy"), means)
    np.save(os.path.join(root_dir_path, "data", data_source, source_dir, model_type, "confidence_region.npy"), confidence_region)

    os.makedirs(os.path.join(root_dir_path, "figs", data_source, source_dir, model_type, "model_fits"), exist_ok=True)
    user_is = sorted(list(set([i for i, k in model_map])))
    if len(user_is) > 100:
        user_is = user_is[:100]
    for i in tqdm.tqdm(user_is, desc="Plotting model fits"):
        user_idxs = [idx for idx in range(len(model_map)) if model_map[idx][0] == i]
        user_ks = [model_map[idx][1] for idx in user_idxs]
        fig, axes = plt.subplots(nrows=len(user_idxs), ncols=1, figsize=(8, 3*len(user_idxs)))
        for idx, (model_idx, k) in enumerate(zip(user_idxs, user_ks)):
            if len(user_idxs) == 1:
                ax = axes
            else:
                ax = axes[idx]
            opinion_name = stance_names[k]
            model = models[model_idx]
            train_x = X[i, ~torch.isnan(y[i,:,k])]
            train_y = y[i, ~torch.isnan(y[i,:,k]), k]
            if model_type == 'gp':
                likelihood = likelihood_list[model_idx]
                plot(model, likelihood, train_x=train_x, train_y=train_y, plot_observed_data=True, plot_predictions=True, ax=ax, opinion_name=opinion_name, plot_legend=idx==0)
            elif model_type == 'spline':
                timeline, mean, confidence_interval = timestamps[i,:,k], means[i,:,k], confidence_region[i,:,k]
                plot_spline(timeline, mean, confidence_interval, train_x=train_x, train_y=train_y, plot_observed_data=True, plot_predictions=True, ax=ax, opinion_name=opinion_name, plot_legend=idx==0)
        # fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(root_dir_path, "figs", data_source, source_dir, model_type, "model_fits", f"{i}_gp.png"))
        plt.close(fig)


            
if __name__ == '__main__':
    main()

