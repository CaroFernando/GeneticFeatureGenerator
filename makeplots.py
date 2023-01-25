import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def make_boxplots(tests, save_path):
    mses = pd.DataFrame(columns = tests.keys())
    # maes = pd.DataFrame(columns = tests.keys())
    r2s = pd.DataFrame(columns = tests.keys())
    new_mses = pd.DataFrame(columns = tests.keys())
    # new_maes = pd.DataFrame(columns = tests.keys())
    new_r2s = pd.DataFrame(columns = tests.keys())

    for key in tests.keys():
        mses[key] = tests[key]["MSE"]
        # maes[key] = tests[key]["MAE"]
        r2s[key] = tests[key]["R2"]
        new_mses[key] = tests[key]["NEW_MSE"]
        # new_maes[key] = tests[key]["NEW_MAE"]
        new_r2s[key] = tests[key]["NEW_R2"]

    NO_MODELS = len(tests.keys())
    model_names = [i for i in tests.keys()]
    
    filter(lambda name: name != 'SVR', model_names)
    
    fig, axs = plt.subplots(NO_MODELS, 2, figsize=(10, 20))
    # Change title font size
    plt.rc('figure', titlesize=40)

    fig.suptitle(f'{save_path} dataset results'.capitalize(), x=0.5, y=0.965)

    fig.tight_layout(pad=6.0)

    title_font_size = 22
    label_font_size = 22

    for ax in axs:
        ax[0].tick_params(axis='both', which='major', labelsize=18)
        ax[1].tick_params(axis='both', which='major', labelsize=18)

    for i in range(NO_MODELS):

        sns.set_theme(style="white", palette=None)

        old_new_mse = pd.DataFrame({
            "MSE" : np.concatenate((mses.iloc[:, i], new_mses.iloc[:, i])),
            "Data" : np.concatenate((np.repeat("Old", mses.shape[0]), np.repeat("New", new_mses.shape[0]))),
            "Model" : np.repeat(model_names[i], mses.shape[0] + new_mses.shape[0])
        })

        b = sns.boxplot(x="MSE", y="Data", data=old_new_mse, ax=axs[i, 0], orient="h", width=0.5)
        b.axes.set_title(model_names[i] + ' MSE',fontsize=title_font_size)
        b.set_xlabel('MSE', fontsize=label_font_size)
        b.set_ylabel('MSE value', fontsize=label_font_size)


        old_new_r2 = pd.DataFrame({
            "R2" : np.concatenate((r2s.iloc[:, i], new_r2s.iloc[:, i])),
            "Data" : np.concatenate((np.repeat("Old", r2s.shape[0]), np.repeat("New", new_r2s.shape[0]))),
            "Model" : np.repeat(model_names[i], r2s.shape[0] + new_r2s.shape[0])
        })

        b = sns.boxplot(x="R2", y="Data", data=old_new_r2, ax=axs[i, 1], orient="h", width=0.5)
        b.axes.set_title(model_names[i] + ' R2',fontsize=title_font_size)
        b.set_xlabel('R2', fontsize=label_font_size)
        b.set_ylabel('R2 value', fontsize=label_font_size)

    # save plot
    plt.savefig(f"plots/{save_path}.png")


def generate_plot_from_csv(path, name):
    print('Here')
    models = ['GradientBoostingRegressor', 'MLPRegressor', 'RandomForestRegressor', 'SGDRegressor']
    tests = dict()
    for model_name in models:
        tests[model_name] = pd.read_csv(f"{path}/{name}/{model_name}.csv")
    make_boxplots(tests, name)
    return

if __name__ == "__main__":
    generate_plot_from_csv("results", 'diabetes')
    generate_plot_from_csv("results", 'insurance')
    generate_plot_from_csv("results", 'wine')
    generate_plot_from_csv("results", 'university')

