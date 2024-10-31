import matplotlib.pyplot as plt
import re
import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import glob

from dassl.utils import check_isfile, listdir_nohidden


def parse_function(*metrics, directory="", pattern=None, end_signal=None):
    subdirs = listdir_nohidden(directory, sort=True)

    outputs = []

    for subdir in subdirs:
        fpath = osp.join(directory, subdir, "log.txt")
        if pattern:
            fpath = glob.glob(osp.join(directory, subdir, pattern))[-1]
        assert check_isfile(fpath)
        results = False
        output = OrderedDict()

        with open(fpath, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()

                if line == end_signal:
                    results = True
                    continue

                for metric in metrics:
                    match = metric["regex"].search(line)
                    if match and results:
                        if "file" not in output:
                            output["file"] = fpath
                        num = float(match.group(1))
                        name = metric["name"]
                        output[name] = num
                    continue

        if output:
            outputs.append(output)

    assert len(outputs) > 0, f"Nothing found in {directory}"

    metrics_results = defaultdict(list)
    for output in outputs:
        msg = ""
        for key, value in output.items():
            if isinstance(value, float):
                msg += f"{key}: {value:.1f}%. "
            else:
                msg += f"{key}: {value}. "
            if key != "file":
                metrics_results[key].append(value)

    output_results = OrderedDict()
    for key, values in metrics_results.items():
        avg = np.mean(values)
        std = np.std(values)
        output_results[key] = (avg, std)

    return output_results


def parse_function_1ls(*metrics, directory="", pattern=None, end_signal=None):
    subdirs = listdir_nohidden(directory, sort=True)

    outputs = []

    for subdir in subdirs:
        for subsubdir in listdir_nohidden(osp.join(directory, subdir), sort=True):
            fpath = osp.join(directory, subdir, subsubdir, "log.txt")
            if pattern:
                fpath = glob.glob(osp.join(directory, subdir, subsubdir, pattern))[-1]
            assert check_isfile(fpath)
            results = False
            output = OrderedDict()

            with open(fpath, "r") as f:
                lines = f.readlines()

                for line in lines:
                    line = line.strip()

                    if line == end_signal:
                        results = True
                        continue

                    for metric in metrics:
                        match = metric["regex"].search(line)
                        if match and results:
                            if "file" not in output:
                                output["file"] = fpath
                            num = float(match.group(1))
                            name = metric["name"]
                            output[name] = num
                        continue

            if output:
                outputs.append(output)

    assert len(outputs) > 0, f"Nothing found in {directory}"

    metrics_results = defaultdict(list)
    for output in outputs:
        msg = ""
        for key, value in output.items():
            if isinstance(value, float):
                msg += f"{key}: {value:.1f}%. "
            else:
                msg += f"{key}: {value}. "
            if key != "file":
                metrics_results[key].append(value)

    output_results = OrderedDict()
    for key, values in metrics_results.items():
        avg = np.mean(values)
        std = np.std(values)
        output_results[key] = (avg, std)

    return output_results


def main(keyword, parent_directory, pattern, end_signal):
    m = {
        "name": keyword,
        "regex": re.compile(fr"\* {keyword}: ([\.\deE+-]+)%"),
    }
    final_results = defaultdict(list)

    for directory in listdir_nohidden(parent_directory, sort=True):
        directory = osp.join(parent_directory, directory)
        results = parse_function(
            m, directory=directory, pattern=pattern, end_signal=end_signal
        )

        for key, value in results.items():
            final_results[key].append(value[0])

    for key, values in final_results.items():
        avg = np.mean(values)
        std = np.std(values)
    
    return avg, std


def read_and_plot(dataset, nlab, baseline, imbalance, targets, weight_h, me,  verbose=False):
    keyword = "accuracy"
    pattern = "log.txt"
    end_signal = "=> result"
    metric = {
        "name": keyword,
        "regex": re.compile(fr"\* {keyword}: ([\.\deE+-]+)%"),
    }
    results = {}
    for t in targets:
        if verbose:
            print(f"Target: {t}")
        for wh in weight_h:
            try:
                if wh == 0:
                    parent_directory = f"../output/ssdg_{dataset}/nlab_{nlab}/{baseline}/{imbalance}/ME/baseline/batchsize_48/ME/resnet18/{t}"
                else:
                    parent_directory = f"../output/ssdg_{dataset}/nlab_{nlab}/{baseline}/{imbalance}/ME/{me}_{wh}/batchsize_48/ME/resnet18/{t}"
                avg, std = parse_function(metric, directory=parent_directory, pattern=pattern, end_signal=end_signal)[keyword]
                results[(wh, t)] = (avg, std)
                if verbose:
                    print(f"Average {keyword} for {me}_{wh}: {avg:.1f}% +- {std:.1f}%")
            except:
                if verbose:
                    print(f"{parent_directory} not found")
                continue
        if verbose:
            print('-----------------------------------------------------------')

    # Create a single subplot for all targets
    fig, ax = plt.subplots(1, len(targets), figsize=(4 * len(targets), 5), sharey=True)
    fig.suptitle(f'{baseline} - {me} - {imbalance} - {dataset}')

    # Iterate over targets
    for i, target in enumerate(targets):
        # Plot the baseline
        try:
            ax[i].axhline(y=results[(0, target)][0], linestyle='--', label=baseline)
        except KeyError:
            pass
        x_values = []
        y_values = []
        std_values = []
        
        # Iterate over alphas
        for wh in weight_h[1:]:
            try:
                # Get the average and std for the current lambda and alpha
                avg, std = results[(wh, target)]
                x_values.append(wh)
                y_values.append(avg)
                std_values.append(std)
            except KeyError:
                continue
        
        # Plot the line and fill_between for the current lambda in the corresponding subplot
        if me=='alpha':
            label = f'{baseline} + H_α'
        else:
            label = f'{baseline} + λ*H'
        ax[i].plot(x_values, y_values, marker='o', label=label, color='blue')
        ax[i].fill_between(x_values, [y - std for y, std in zip(y_values, std_values)], [y + std for y, std in zip(y_values, std_values)], alpha=0.2, color='blue')
    
        ax[i].set_xlabel('Alpha' if me == 'alpha' else 'Lambda')
        if i == 0:
            ax[i].set_ylabel('Accuracy (%)')
        ax[i].yaxis.set_tick_params(labelleft=True)
        ax[i].set_title(target)
        ax[i].legend()

    plt.tight_layout()
    plt.show()