import matplotlib.pyplot as plt

TOKENS_PER_STEP = 524288

if __name__ == '__main__':
    logfiles = [
        'pylog124M/e9067b07-f2e0-4b0d-bfbe-ca25a22e60f0.log', # vanilla, fineweb
        # 'pylog124M/ee8b9c4c-9630-4651-9b22-1b26c2ec7e75.log', # vanilla, fineweb-edu
        # 'pylog124M_l1_loss/0a4aeefd-0601-49a4-a178-f2ad7b6e1a8b.log', # fineweb-edu l = 1.0 on mlp out, ReLU
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
