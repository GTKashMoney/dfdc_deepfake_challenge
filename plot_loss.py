import argparse

import numpy as np

from matplotlib import pyplot as plt


def parse_losses(logs):
    real_losses = {str(int(i)+1): [] for i in range(6)}
    fake_losses = {str(int(i)+1): [] for i in range(6)}
    for log in logs:
        log = log.replace('.0_','_')
        if 'type' in log and 'loss' in log:
            if 'fake_loss' in log:
                fk_loss = float(log.split(' ')[-1].replace('\n', ''))
                fake_losses[log.split('_')[0][-1]].append(fk_loss)
        
            if 'real_loss' in log:
                rl_loss = float(log.split(' ')[-1].replace('\n', ''))
                real_losses[log.split('_')[0][-1]].append(rl_loss)
                
    return real_losses, fake_losses


def get_std(real_losses, fake_losses):
    real_arr = np.nan_to_num(np.array(list(real_losses.values())), 0)
    fake_arr = np.nan_to_num(np.array(list(fake_losses.values())), 0)
    return np.nanstd((real_arr + fake_arr) / 2, axis=0)


def get_mean(real_losses, fake_losses):
    real_arr = np.nan_to_num(np.array(list(real_losses.values())), 0)
    fake_arr = np.nan_to_num(np.array(list(fake_losses.values())), 0)
    return np.nanmean((real_arr + fake_arr) / 2, axis=0)


def get_max(real_losses, fake_losses):
    real_arr = np.nan_to_num(np.array(list(real_losses.values())), 0)
    fake_arr = np.nan_to_num(np.array(list(fake_losses.values())), 0)
    return np.max((real_arr + fake_arr) / 2, axis=0)


def get_min(real_losses, fake_losses):
    real_arr = np.nan_to_num(np.array(list(real_losses.values())), 0)
    fake_arr = np.nan_to_num(np.array(list(fake_losses.values())), 0)
    return np.max((real_arr + fake_arr) / 2, axis=0)


def compare_loss(file_names: list, metric: str = 'std'):
    
    for filename in file_names:
        with open(f'logs/{filename}.txt') as f:
            logs = f.readlines()
        
        real_losses, fake_losses = parse_losses(logs)
        
        if metric == 'std':
            y = get_std(real_losses, fake_losses)
        elif metric == 'mean':
            y = get_mean(real_losses, fake_losses)
        elif metric == 'min':
            y = get_min(real_losses, fake_losses)
        elif metric == 'max':
            y = get_max(real_losses, fake_losses)
        
        plt.plot(list([i + 1 for i in range(len(y))]), y, label=filename)
    
    plt.legend()
    plt.ylabel(f'{metric} of Loss Across Skin Tags')
    plt.xlabel('Epoch')
    plt.title(f'Validation: {metric} loss')
    
    return plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot losses from log")
    parser.add_argument("--log-file", help="path to log file", required=True)
    parser.add_argument("--fake-weight", help="weight for fake loss", default=1.4, type=float)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.log_file, "r") as f:
        lines = f.readlines()
    # real_losses = []
    # fake_losses = []
    max_losses = []
    net_losses = []
    for line in lines:
        line = line.strip()
        if line.startswith("Max"):
            max = float((line.split(",")[0]).split(" ")[-1])
            min = float((line.split(",")[1]).split(" ")[-1])
            max_losses.append(max)
            net_losses.append(max - min)
    #     if line.startswith("fake_loss"):
    #         fake_losses.append(float(line.split(" ")[-1]))
    #     elif line.startswith("real_loss"):
    #         real_losses.append(float(line.split(" ")[-1]))
    # real_losses = np.array(real_losses)
    # fake_losses = np.array(fake_losses)
    max_losses = np.array(max_losses)
    net_losses = np.array(net_losses)
    # loss = (fake_losses * args.fake_weight + real_losses)/2
    # loss = max_losses
    loss = net_losses
    # plt.title("Weighted loss ({}*fake_loss + real_loss)/2)".format(args.fake_weight))
    # plt.title("Max loss over skin types")
    plt.title("Net loss over skin types")
    best_loss_idx = np.argsort(loss)[:5]
    # ignore early epochs  loss is quite noisy and there could be spikes
    best_loss_idx = best_loss_idx[best_loss_idx > 16]
    plt.scatter(best_loss_idx, loss[best_loss_idx], c="red")
    for idx in best_loss_idx:
        plt.annotate(str(idx), (idx, loss[idx]))
    plt.plot(loss)
    plt.show()


if __name__ == '__main__':
    main()
