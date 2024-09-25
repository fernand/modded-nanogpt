import matplotlib.pyplot as plt

TOKENS_PER_STEP = 524288

if __name__ == '__main__':
    logfiles = [
        'pylog124M/vanilla.log', # vanilla, fineweb
        'pylog768_fjlt/764f763b-3303-4c92-a170-cdc541eaafd4.log', # FJLT N=1024 with q but row wise sparsity and norm
        'pylog768_fjlt/926ead4c-003e-470a-8960-d0db61716e2a.log', # FJLT N=1024 with q but row wise sparsity and norm, N(0, 0.02) init
        'pylog768_fjlt/bc95287f-844b-4798-8883-62aa164e92f8.log', # FJLT N=1600 sparsity 383 with q but row wise sparsity and norm, N(0, 0.02) init
    ]
    plot_eval = False
    plot_start = True
    xs, ys = [], []
    for logfile in logfiles:
        x, y = [], []
        with open(logfile) as f:
            for line in f:
                if (plot_eval and 'tel' in line) or (not plot_eval and 'trl' in line):
                    parts = line.strip().split()
                    x.append(int(TOKENS_PER_STEP * int(parts[0][2:]) / 1e6))
                    y.append(float(parts[1][4:]))
        xs.append(x)
        ys.append(y)

    for x, y in zip(xs, ys):
        if plot_eval:
            if plot_start:
                plt.plot(x[:10], y[:10])
            else:
                plt.plot(x[10:], y[10:])
        else:
            if plot_start:
                plt.plot(x[:500], y[:500])
            else:
                plt.plot(x[500:], y[500:])
    plt.xlabel('Tokens')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig('loss.png')
