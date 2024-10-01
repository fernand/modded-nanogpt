import matplotlib.pyplot as plt

TOKENS_PER_STEP = 524288

def plot_loss(curr_logfile):
    logfiles = [
        'pylog124M/vanilla.log', # vanilla dim 768, fineweb
    ]
    logfiles.append(curr_logfile)
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
    plt.close()
