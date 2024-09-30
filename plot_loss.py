import matplotlib.pyplot as plt

TOKENS_PER_STEP = 524288

if __name__ == '__main__':
    logfiles = [
        'pylog124M/vanilla.log', # vanilla dim 768, fineweb
        'pylog_fjlt/3d3148e8-eccc-4967-a28d-91ab1857a313.log', # Random weights
        'pylog_fjlt/d6ddd722-f3a1-4e6d-8829-f74c27794558.log', # 8096 frozen
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
