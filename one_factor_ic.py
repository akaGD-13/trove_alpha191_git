import pandas as pd
import matplotlib.pyplot as plt


def one_factor_ic(prices: pd.DataFrame, factor: pd.DataFrame, factor_name: str, output_path: str) -> float:
    # calcualte ic
    # draw ic
    plt.savefig(output_path + '/ic_graph_' + factor_name)
    ic = 0
    return ic

