import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style(style='white')


def plot_blocking():
    # data = pd.read_csv("../data/non_interaction_case_data_1000_2_3.csv")
    data = pd.read_csv("../data/non_interaction_case_data_10000_1_3.csv")

    # data.plot(x='alpha', y='derivative_energy')
    data.plot(x='alpha', y='variance')
    # data.plot(x='alpha', y=['derivative_energy', 'local_energy'])
    plt.show()


def plot_brute_force():
    data1 = pd.read_csv("../data/Brute-force/non_interaction_brute_force_importance.csv")
    data2 = pd.read_csv("../data/Brute-force/non_interaction_brute_force_metropolis.csv")

    x = data1['alpha']
    y1 = data1['variance']
    y2 = data2['variance']
    # y = data['local_energy']
    # y = data['derivative_energy']
    # e = np.zeros(len(x))
    # plt.errorbar(x, y, yerr=y)
    # plt.plot(x, y)

    fig, ax = plt.subplots()
    ax.plot(x, y1, color='darkorange', linewidth=2.5)
    # ax.plot(x, y2, color='mediumslateblue', linewidth=1.5)
    plt.grid(color='gray', linestyle='-', linewidth=0.2)
    # plt.hlines(1.5, 0, 1)
    # plt.title("Variance of "  r"$\langle E_L \rangle$")
    # plt.title("One particle in 3D, with metropolis sampling")
    # legend((""), loc='upper right')
    plt.xlabel(r"$\alpha$")

    # plt.ylabel(r"$\langle E_L \rangle$")
    plt.ylabel(r"$\sigma^2$")
    # axis([100, 10000, 0, 0.005])

    zoom = True
    """Plot zoomed part where error is smallest"""
    if zoom is True:
        axins = zoomed_inset_axes(ax, 3.5, loc=1)
        axins.plot(x, y1, color='darkorange', linewidth=2.5)
        axins.plot(x, y2, color='mediumslateblue', linewidth=1.5)
        axins.set_xlim(0.45, 0.55)
        axins.set_ylim(-0.5, 0.5)
        plt.yticks(visible=False)
        plt.xticks(visible=False)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.show()


def plot_non_interaction():
    # data = pd.read_csv("../data/non_interaction_brute_force_importance.csv")
    # data = pd.read_csv("../data/non_interaction_case_data_1000_2_3_variance.csv")
    # data = pd.read_csv("../data/Non-interaction/firstrunjanuary/non_interaction_case_data_ana_metro_10000_1_3.csv")
    data = pd.read_csv("../data/non_interaction_case_data.csv")

    x = data['alpha']
    # y = data['variance']
    # y = data['derivative_energy']
    y = data['local_energy']
    h = 1.5

    # e = np.zeros(len(x))
    # plt.errorbar(x, y, yerr=y)
    # plt.plot(x, y)

    fig, ax = plt.subplots()
    ax.plot(x, y, color='yellowgreen', linewidth=2.0)
    plt.grid(color='gray', linestyle='-', linewidth=0.2)
    plt.hlines(h, 0.4, 0.5, color='dimgray')
    # plt.title("Variance of "  r"$\langle E_L \rangle$ " "metropolis analytical")
    # plt.title("One particle, 3D, importance sampling and numerical expr. " r"$E_L$")
    # plt.title("One particle, 3D, metropolis sampling and numerical expr. " r"$E_L$")
    # legend((""), loc='upper right')
    plt.xlabel(r"$\alpha$")

    plt.ylabel(r"$\langle E_L \rangle$")
    # plt.ylabel(r"$\sigma^2$")
    plt.axis([0.5, 0.7, 1.2, 2.4])

    zoom = False
    """Plot zoomed part where error is smallest"""
    if zoom is True:
        axins = zoomed_inset_axes(ax, 2.5, loc=1)
        axins.plot(x, y)
        axins.set_xlim(0.45, 0.55)
        axins.set_ylim(-0.5, 0.5)
        plt.yticks(visible=False)
        plt.xticks(visible=False)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.show()


def plot_weak_interaction():
    # data = pd.read_csv("../data/non_interaction_brute_force_importance.csv")
    # data = pd.read_csv("../data/non_interaction_case_data_1000_2_3_variance.csv")
    data = pd.read_csv("../data/weak_interaction_case_data.csv")

    x = data['alpha']
    y = data['variance']
    # y = data['derivative_energy']
    # y = data['local_energy']
    h = 3.00345

    # e = np.zeros(len(x))
    # plt.errorbar(x, y, yerr=y)
    # plt.plot(x, y)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    # plt.hlines(h, 0.4, 0.5)
    # plt.title("Variance of "  r"$\langle E_L \rangle$ " "metropolis analytical")
    plt.title("Two particles with weak interaction, importance sampling")
    # plt.title("One particle, 3D, metropolis sampling and numerical expr. " r"$E_L$")
    # legend((""), loc='upper right')
    plt.xlabel(r"$\alpha$")

    plt.ylabel(r"$\langle E_L \rangle$")
    # plt.ylabel(r"$\sigma^2$")
    # axis([100, 10000, 0, 0.005])

    zoom = False
    """Plot zoomed part where error is smallest"""
    if zoom is True:
        axins = zoomed_inset_axes(ax, 2.5, loc=1)
        axins.plot(x, y)
        axins.set_xlim(0.45, 0.55)
        axins.set_ylim(-0.5, 0.5)
        plt.yticks(visible=False)
        plt.xticks(visible=False)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.show()


def plot_one_body_density():
    # data = pd.read_csv("../data/non_interaction_case_data_1000_2_3.csv")
    data = pd.read_csv("../data/obd_data.csv")

    x = data['r'].to_numpy()
    y = data['density'].to_numpy()
    print (type(y))
    # data.plot(x='alpha', y='derivative_energy')
    b = int(len(x)/4)
    # plt.hist(y, bins=b)
    sns.distplot(y)
    # data.plot(x='alpha', y=['derivative_energy', 'local_energy'])
    plt.show()


def plot_interation():
    data1 = pd.read_csv("../data/non_interaction_gradient_descent_0.01.csv")
    data2 = pd.read_csv("../data/non_interaction_gradient_descent_0.1.csv")
    data3 = pd.read_csv("../data/non_interaction_gradient_descent_BB.csv")

    x = data1['iterations']
    x_BB = data3['iterations']
    # y = data['alpha']
    # y = data['variance']
    # y = data['derivative_energy']
    y1 = data1['local_energy']
    y2 = data2['local_energy']
    y3 = data3['local_energy']
    h = 1.5

    fig, ax = plt.subplots()
    # ax.plot(x, y1, y2, y3)
    plt.plot(x_BB, y3, label="BB", color='yellowgreen', linewidth=2.0)
    plt.plot(x, y2, label="GD 0.1", color='mediumpurple', linewidth=1.6)
    plt.plot(x, y1, label="GD 0.01", color='darkorange', linewidth=1.2)
    plt.grid(color='gray', linestyle='-', linewidth=0.2)
    # plt.hlines(h, 0, 100)
    # plt.title("Variance of "  r"$\langle E_L \rangle$ " "metropolis analytical")
    # plt.title("One particle, 3D, importance sampling and numerical expr. " r"$E_L$")
    # plt.title("One particle, 3D, metropolis sampling with 10000 mcc")
    # legend((""), loc='upper right')
    plt.legend(loc='upper right')
    plt.xlabel("Iterations")

    plt.ylabel(r"$\langle E_L \rangle$")
    # plt.ylabel(r"$\sigma^2$")
    # axis([100, 10000, 0, 0.005])

    zoom = False
    """Plot zoomed part where error is smallest"""
    if zoom is True:
        axins = zoomed_inset_axes(ax, 2.5, loc=1)
        axins.plot(x, y)
        axins.set_xlim(0.45, 0.55)
        axins.set_ylim(-0.5, 0.5)
        plt.yticks(visible=False)
        plt.xticks(visible=False)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.show()


# plot_interation()
plot_non_interaction()
# plot_weak_interaction()
# plot_one_body_density()
# plot_blocking()
# plot_brute_force()
