import matplotlib.pyplot as plt

TOKENS_PER_STEP = 524288

if __name__ == '__main__':
    logfiles = [
        'pylog124M/vanilla.log', # vanilla dim 768, fineweb
        'pylog768_fjlt/deedd85a-3f10-4a0f-9aa4-9de1b0211629.log', # FJLT N=32768 sparsity 16 separate random proj per MLP frozen MLP
        'pylog_fjlt/410f157d-9aee-40f0-aa3f-8e1e18f088d7.log', # FJLT N=32768 sparsity 384 separate random proj per MLP
        'pylog_fjlt/9c56c141-e3c2-4e79-b33e-40929a70ef3c.log', # sparsity 384, fix init
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
