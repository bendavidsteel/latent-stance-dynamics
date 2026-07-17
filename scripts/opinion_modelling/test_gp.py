import datetime
import dotenv
import os
import re

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import requests
import torch
from torch import multiprocessing
from tqdm import tqdm
import wandb

from mining.estimate import setup_ordinal_gp_model, train_ordinal_likelihood_gp, get_model_prediction, get_likelihood_prediction

def plot_gp(
    train_x,
    train_y,
    test_x,
    pred,
    lower,
    upper,
    observed_pred
):
    # Get into evaluation (predictive posterior) mode
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 4))
    ax, ax_likelihood = axes
    
    # Plot training data as black stars
    ax.plot(train_x, train_y, 'k*')
    
    # Plot predictive means as blue line
    ax.plot(test_x, pred, 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x, lower, upper, alpha=0.5)
    
    ax_likelihood.plot(train_x, train_y, 'k*')
    test_x = torch.tensor(test_x, dtype=torch.float32, device='cuda')
    
    # average over samples
    map = ax_likelihood.matshow(
        observed_pred.probs.mean(0).cpu().numpy().T, 
        aspect='auto', 
        extent=(train_x.min(), train_x.max(), -1.5, 1.5), 
        origin='lower'
    )
    fig.colorbar(map, ax=ax_likelihood)

    ax.hlines(y=np.mean(train_y), xmin=train_x.min(), xmax=train_x.max(), color='red')

    ax.set_ylim([-1.5, 1.5])
    ax.legend(['Observed Data', 'GP Mean', 'Confidence', 'Mean'])
    return fig

def get_session(base_url, username, password):
    res = requests.post(f"{base_url}/meologin", params={"username": username, "password": password}, verify=True)
    token = res.json()["access_token"]
    headers = {
        "Authorization": f"Bearer {token}"
    }
    return headers

def get_seedlist(base_url, headers):
    target_url = f"{base_url}/phh/seedlist/"
    res = requests.get(target_url, params={"query": '*'}, headers=headers)
    if res.status_code == 200:
        return res.json()
    else:
        return None

def process_data():
    """Load and process the main data from files using efficient Polars operations"""
    print("Loading and processing data files...")
    # Load data with fewer columns
    dir_path = './data/stance_targets/doc_stance'
    
    # Collect all file paths first, then read them in a single operation
    file_paths = [
        os.path.join(dir_path, file) 
        for file in os.listdir(dir_path)
        if re.search(r'\d{4}_\d{1,2}_doc_targets_with_stance.parquet.zstd', file)
    ]

    # file_paths = sorted(file_paths, reverse=True)[:20] # TODO remove for full version
    
    if not file_paths:
        raise ValueError("No stance data files found in the data directory")
    
    dfs = [
        pl.read_parquet(
            file_path,
            columns=['id', 'createtime', 'platform', 'Document', 'ParentDocument', 'Targets', 'Polarities', 'seed']
        ) for file_path in file_paths
    ]
    
    # Concatenate the scanned DataFrames
    df = pl.concat(dfs, how='diagonal_relaxed')

    df = df.unique(['id', 'platform'])
    
    df = df.explode(['Targets', 'Polarities'])\
        .rename({'Targets': 'Target', 'Polarities': 'Stance'})

    df = df.drop_nulls('Target')
    # df = df.filter(pl.col('Target').str.contains_any(['israel', 'gaza', 'ukraine', 'russia', 'china', 'carbon tax', 'immigration']))

    
    # Get ordered list of all targets in a single pipeline
    target_count_df = df.group_by('Target').agg(pl.count().alias('count')).sort('count', descending=True)
    
    return df, target_count_df

def get_timestamps(df, start_date):
    return df.select(((pl.col('createtime') - start_date).dt.total_hours() / (24 * 30)).alias('timestamps'))['timestamps'].to_numpy()


def remove_low_count_targets(df, target_count_df, min_count):
    removed_target_df = target_count_df.filter(pl.col('count') < min_count)
    target_count_df = target_count_df.filter(pl.col('count') >= min_count)
    df = df.join(removed_target_df, on='Target', how='anti')
    return df, target_count_df

def get_classifier_profiles():
    api = wandb.Api()
    project_name = os.environ['PROJECT_NAME']
    runs = api.runs(project_name)
    
    model_name = 'HuggingFaceTB/SmolLM2-135M-Instruct'
    classification_method = 'head'
    dataset = ['vast', 'ezstance', 'ezstance_claim', 'pstance', 'semeval', 'mtcsd']
    
    filtered_runs = [
        run for run in runs
        if run.state == "finished"\
            and 'test/confusion_matrix' in run.summary\
            and run.config['model_name'] == model_name\
            and run.config['classification_method'] == classification_method\
            and run.config['dataset'] == dataset
    ]
    filtered_runs = sorted(filtered_runs, key=lambda run: datetime.datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%SZ"))

    confusion_matrix = filtered_runs[-1].summary.get('test/confusion_matrix')

    labels2id = {
        "neutral": 0,
        "favor": 1,
        "against": 2
    }

    classifier_profiles = {
        0: {
            'true_against': {
                'predicted_against': confusion_matrix[labels2id['against']][labels2id['against']],
                'predicted_neutral': confusion_matrix[labels2id['against']][labels2id['neutral']],
                'predicted_favor': confusion_matrix[labels2id['against']][labels2id['favor']]
            },
            'true_neutral': {
                'predicted_against': confusion_matrix[labels2id['neutral']][labels2id['against']],
                'predicted_neutral': confusion_matrix[labels2id['neutral']][labels2id['neutral']],
                'predicted_favor': confusion_matrix[labels2id['neutral']][labels2id['favor']]
            },
            'true_favor': {
                'predicted_favor': confusion_matrix[labels2id['favor']][labels2id['favor']],
                'predicted_neutral': confusion_matrix[labels2id['favor']][labels2id['neutral']],
                'predicted_against': confusion_matrix[labels2id['favor']][labels2id['against']]
            }
        }
    }
    # classifier_profiles = {
    #     0: {
    #         'true_against': {
    #             'predicted_against': 7,
    #             'predicted_neutral': 0,
    #             'predicted_favor': 0
    #         },
    #         'true_neutral': {
    #             'predicted_against': 0,
    #             'predicted_neutral': 7,
    #             'predicted_favor': 0
    #         },
    #         'true_favor': {
    #             'predicted_favor': 7,
    #             'predicted_neutral': 0,
    #             'predicted_against': 0
    #         }
    #     }
    # }
    return classifier_profiles

def get_time_series_data(filtered_df):
    sorted_df = filtered_df.sort('createtime')
    start_date = filtered_df['createtime'].min().date()
    end_date = filtered_df['createtime'].max().date()
    timestamps = get_timestamps(sorted_df, start_date)

    days = []
    current_date = start_date
    while current_date <= end_date:
        days.append(current_date)
        current_date += datetime.timedelta(days=1)
    day_df = pl.DataFrame({'createtime': days})

    # Calculate volume
    day_df = day_df.join(
            sorted_df.select(pl.col('createtime').dt.date())\
                .group_by('createtime')\
                .len()\
                .rename({'len': 'volume'}),
            on='createtime',
            how='left'
        )\
        .fill_null(0)\
        .group_by(pl.col('createtime').dt.truncate('1d'))\
        .agg(pl.col('volume').sum())\
        .sort('createtime')

    test_x = get_timestamps(day_df, start_date)
    
    stance = sorted_df['Stance'].to_numpy()

    classifier_ids = np.zeros_like(timestamps, dtype=np.uint16)

    return timestamps, stance, classifier_ids, test_x

def get_timeseries(args):
    timestamps, stance, classifier_ids, classifier_profiles, lengthscale_loc, lengthscale_scale, sigma_loc, sigma_scale, test_x = args
    
    model, likelihood, train_x, train_y, classifier_ids = setup_ordinal_gp_model(
        timestamps, 
        stance, 
        classifier_ids, 
        classifier_profiles, 
        lengthscale_loc,
        lengthscale_scale,
        sigma_loc=sigma_loc,
        sigma_scale=sigma_scale
    )
    model, likelihood, losses = train_ordinal_likelihood_gp(model, likelihood, train_x, train_y, classifier_ids, verbose=False)
    pred, lower, upper = get_model_prediction(model, test_x)
    observed_pred = get_likelihood_prediction(model, likelihood, test_x)

    lengthscale = model.covar_module.base_kernel.lengthscale.item()
    likelihood_sigma = likelihood.sigma.item()
    
    return lengthscale, likelihood_sigma, pred, lower, upper, observed_pred

def main():
    dotenv.load_dotenv()
    pl.set_random_seed(42)

    df, target_count_df = process_data()

    # remove targets with low counts
    df, target_count_df = remove_low_count_targets(df, target_count_df, 50)

    # target_idx = 0
    # target_name = target_count_df['Target'][target_idx]
    target_name = 'trudeau'
    target_df = df.filter(pl.col('Target') == target_name)
    
    # filtered_df = target_df
    platform_handle_idx = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    top_user_df = target_df.select(pl.col('seed').struct.field('PlatformHandleID')).group_by('PlatformHandleID').len().filter(pl.col('len') >= 5)['PlatformHandleID']
    platform_handle_ids = top_user_df[platform_handle_idx]
    # platform_handle_id = '5866-Twitter-DeniseInCanada'

    target_df = target_df.filter(pl.col('seed').struct.field('PlatformHandleID').is_in(top_user_df))
    filtered_df = target_df.filter(pl.col('seed').struct.field('PlatformHandleID').is_in(platform_handle_ids))

    all_timestamps, all_stance, all_classifier_ids, all_test_x = [], [], [], []
    for platform_handle_id in platform_handle_ids:
        platform_handle_df = filtered_df.filter(pl.col('seed').struct.field('PlatformHandleID') == platform_handle_id)
        timestamps, stance, classifier_ids, test_x = get_time_series_data(platform_handle_df)
        all_timestamps.append(timestamps)
        all_stance.append(stance)
        all_classifier_ids.append(classifier_ids)
        all_test_x.append(test_x)

    classifier_profiles = get_classifier_profiles()

    # log normal
    # mode at ~3 months
    lengthscale_loc = 2.0
    lengthscale_scale = 0.3

    sigma_loc = 1.0
    sigma_scale = 0.1

    # TODO parallel with multiprocessing https://docs.pytorch.org/docs/stable/notes/multiprocessing.html

    multiprocessing.set_start_method('spawn', force=True)

    args_list = []
    for i in range(len(all_timestamps)):
        args_list.append((all_timestamps[i], all_stance[i], all_classifier_ids[i], classifier_profiles, lengthscale_loc, lengthscale_scale, sigma_loc, sigma_scale, all_test_x[i]))

    start_time = datetime.datetime.now()
    for args in args_list:
        get_timeseries(args)
    end_time = datetime.datetime.now()
    print(f"Sequential duration: {end_time - start_time}")

    start_time = datetime.datetime.now()
    with multiprocessing.Pool(processes=4) as pool:
        results = list(tqdm(pool.imap(get_timeseries, args_list), total=len(args_list), desc='Training GPs:'))
    end_time = datetime.datetime.now()
    print(f"Parallel duration: {end_time - start_time}")

    for result in results:
        lengthscale, likelihood_sigma, pred, lower, upper, observed_pred = result
        print(f"Learned lengthscale: {lengthscale}")
        print(f"Learned sigma: {likelihood_sigma}")

        # get some samples from the untrained models
        fig = plot_gp(
            timestamps, 
            stance, 
            test_x,
            pred,
            lower,
            upper,
            observed_pred
        )
        fig.savefig('./figs/test_gp.png')


if __name__ == "__main__":
    main()