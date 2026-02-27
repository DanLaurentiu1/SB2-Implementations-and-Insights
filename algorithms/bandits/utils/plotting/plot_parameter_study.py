import matplotlib.pyplot as plt
import numpy as np


def plot_parameter_study():
    plt.figure(figsize=(10, 6))
    plt.xscale("log", base=2)
    ticks = [2**i for i in range(-7, 3)]
    labels = ["1/128", "1/64", "1/32", "1/16", "1/8", "1/4", "1/2", "1", "2", "4"]
    plt.xticks(ticks, labels)
    plt.yticks(np.arange(1, 1.5 + 0.1, 0.1))
    plt.xlim(1 / 150, 4)

    plt.ylabel("Average reward over first 1000 steps", fontsize=12)
    plt.xlabel(r"$\epsilon \quad \alpha \quad c \quad Q_0$", fontsize=18)

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    ucb_x = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
    ucb_y = [
        1.3949718617345954,
        1.4099994873013635,
        1.4327476617636719,
        1.462341481228085,
        1.463299936351778,
        1.3727462578576062,
        1.147980868370038,
    ]

    opt_greedy_x = [0.25, 0.5, 1.0, 2.0, 4.0]
    opt_greedy_y = [
        1.31275170925888447,
        1.3827399564251786,
        1.4434084476196273,
        1.3982655709866091,
        1.3151370936057473,
    ]

    eps_greedy_x = [0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25]
    eps_greedy_y = [
        1.1745674931251786,
        1.2085519627399564,
        1.2851004849884617,
        1.3060123105744312,
        1.25293490925914,
        1.111847634084476,
    ]

    gradient_bandit_x = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
    gradient_bandit_y = [
        1.0823278265570955,
        1.271389249916708,
        1.361460090676811,
        1.3919586609110361,
        1.370995460193386,
        1.2378439934670076,
        1.011660327697986,
        0.759853367322699,
    ]

    plt.plot(ucb_x, ucb_y, color="blue")
    plt.plot(
        opt_greedy_x,
        opt_greedy_y,
        color="black",
    )
    plt.plot(eps_greedy_x, eps_greedy_y, color="red")
    plt.plot(
        gradient_bandit_x,
        gradient_bandit_y,
        color="limegreen",
    )

    plt.text(1 / 64, 1.25, r"$\epsilon$-greedy", color="red", fontsize=14, ha="center")
    plt.text(1, 1.05, "gradient\nbandit", color="limegreen", fontsize=14, ha="center")
    plt.text(1 / 4, 1.45, "UCB", color="blue", fontsize=14, ha="center")
    plt.text(
        4.0,
        1.4,
        r"greedy with" "\n" r"optimistic" "\n" r"initialization" "\n" r"$\alpha = 0.1$",
        color="black",
        fontsize=14,
        ha="right",
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_parameter_study()
