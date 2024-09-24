import matplotlib.pyplot as plt

TOKENS_PER_STEP = 524288

if __name__ == '__main__':
    logfiles = [
        'pylog124M/e9067b07-f2e0-4b0d-bfbe-ca25a22e60f0.log',
        'pylog124M_l1_loss/0d496d63-2e59-4655-bd64-ff4ed9d2e5af.log', # l = 1.0 on residual
        'pylog124M_l1_loss/e23cc435-a592-4e2c-8525-22930857215c.log', # l = 1.0 on mlp out
    ]
    plot_eval = False
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

    plt.figure(figsize=(10, 6))
    for x, y in zip(xs, ys):
        plt.plot(x[:500], y[:500])
    plt.xlabel('Tokens')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig('loss.png')
