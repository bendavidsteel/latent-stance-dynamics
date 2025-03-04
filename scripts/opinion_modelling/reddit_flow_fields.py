
import os

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from mining.dynamics import flow_pairplot

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir_path = os.path.join(this_dir_path, "..", "..")

    data_source = "reddit"
    source_dir = "1sub_1year"
    if source_dir == "4sub_1year":
        topics_dir_path = os.path.join(root_dir_path, "data", data_source, source_dir)
    elif source_dir == "1sub_1year":
        topics_dir_path = os.path.join(root_dir_path, "data", data_source, source_dir, "topics_minilm_0_2")
    subsample_users = None
    # dataset = RedditOpinionTimelineDataset(topics_dir_path, subsample_users=subsample_users)

    model_type = 'spline'

    stance_columns = ['vaccine_mandates', 'renter_protections', 'ndp', 'liberals', 'conservatives', 'gun_control', 'drug_decriminalization', 'liberal_immigration_policy', 'canadian_aid_to_ukraine']
    stance_names = ['Vaccine Mandates', 'Renter Protections', 'NDP', 'Liberals', 'Conservatives', 'Gun Control', 'Drug Decriminalization', 'Liberal Immigration Policy', 'Canadian Aid to Ukraine']
    timestamps = np.load(os.path.join(root_dir_path, "data", data_source, source_dir, model_type, "timestamps.npy"))
    means = np.load(os.path.join(root_dir_path, "data", data_source, source_dir, model_type, "means.npy"))
    confidence_intervals = np.load(os.path.join(root_dir_path, "data", data_source, source_dir, model_type, "confidence_region.npy"))


    flow_dir_path = os.path.join(root_dir_path, "figs", data_source, source_dir, model_type, "flows")
    os.makedirs(flow_dir_path, exist_ok=True)

    # TODO ensure timestamps are correctly plotted, the figure has suspiciously many people with similar start times
    figs, axes, flow_figs, flow_axes = flow_pairplot(timestamps, means, confidence_intervals, stance_names, do_pairplot=False, save=True, save_path=flow_dir_path)

    # for i in range(len(axes)):
    #     for j in range(len(axes[i])):
    #         fig = figs[i, j]
    #         fig.savefig(os.path.join(flow_dir_path, f'{i}_{j}_flows.png'))

    # for i in range(len(flow_axes)):
    #     fig = flow_figs[i]
    #     fig.savefig(os.path.join(flow_dir_path, f'{i}_flows.png'))

            
if __name__ == '__main__':
    main()

