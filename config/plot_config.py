import matplotlib.pyplot as plt


def setup_plotting():
    # set plotting parameters
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": False,
            "pgf.rcfonts": False,
            "pgf.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}",
        }
    )

    # set plotting style
    plt.style.use("seaborn-v0_8-colorblind")

    # font size
    plt.rcParams["font.size"] = 12
