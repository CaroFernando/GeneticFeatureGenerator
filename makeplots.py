import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def make_boxplots(tests, save_path):
    mses = pd.DataFrame(columns = tests.keys())
    maes = pd.DataFrame(columns = tests.keys())
    r2s = pd.DataFrame(columns = tests.keys())
    new_mses = pd.DataFrame(columns = tests.keys())
    new_maes = pd.DataFrame(columns = tests.keys())
    new_r2s = pd.DataFrame(columns = tests.keys())

    for key in tests.keys():
        mses[key] = tests[key]["MSE"]
        maes[key] = tests[key]["MAE"]
        r2s[key] = tests[key]["R2"]
        new_mses[key] = tests[key]["NEW_MSE"]
        new_maes[key] = tests[key]["NEW_MAE"]
        new_r2s[key] = tests[key]["NEW_R2"]

    NO_MODELS = len(tests.keys())
    model_names = [i for i in tests.keys()]

    fig, axs = plt.subplots(NO_MODELS, 2, figsize=(10, 20))
    fig.suptitle(f'{save_path} dataset results')

    # add padding
    fig.tight_layout(pad=4.0)

    # plot mse and r2

    for i in range(NO_MODELS):

        sns.set_theme(style="white", palette=None)

        old_new_mse = pd.DataFrame({
            "MSE" : np.concatenate((mses.iloc[:, i], new_mses.iloc[:, i])),
            "Data" : np.concatenate((np.repeat("Old", mses.shape[0]), np.repeat("New", new_mses.shape[0]))),
            "Model" : np.repeat(model_names[i], mses.shape[0] + new_mses.shape[0])
        })

        sns.boxplot(x="MSE", y="Data", data=old_new_mse, ax=axs[i, 0], orient="h", width=0.5)
        # add label for model name
        axs[i, 0].set_title(model_names[i] + ' MSE')
        axs[i, 0].set(ylabel='MSE value')
        # set mse y range between 0 and 62000


        old_new_r2 = pd.DataFrame({
            "R2" : np.concatenate((r2s.iloc[:, i], new_r2s.iloc[:, i])),
            "Data" : np.concatenate((np.repeat("Old", r2s.shape[0]), np.repeat("New", new_r2s.shape[0]))),
            "Model" : np.repeat(model_names[i], r2s.shape[0] + new_r2s.shape[0])
        })

        sns.boxplot(x="R2", y="Data", data=old_new_r2, ax=axs[i, 1], orient="h", width=0.5)

        # add label for model name
        axs[i, 1].set_title(model_names[i] + ' R2')
        axs[i, 1].set(ylabel='R2 value')

    # save plot
    plt.savefig(f"plots/{save_path}.png")
