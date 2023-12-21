import os
from statistics import median
from typing import List, Union

import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame


def line_plot(df: DataFrame, x: str, y: Union[List[str], str], groups: Union[List[str], str] = None,
              file_prefix: str = "", cols_joiner: str = "_", filename_joiner: str = "_"):
    if isinstance(y, str):
        y = [y]
    if isinstance(groups, str):
        groups = [groups]

    def q1(a):
        return a.quantile(0.25)

    def q3(b):
        return b.quantile(0.75)

    vals = dict([(key, [q1, q3, median]) for key in y])

    summary = df.groupby(groups + [x]).agg(vals)
    summary.columns = [cols_joiner.join(col) for col in summary.columns.to_flat_index()]
    summary.reset_index(inplace=True)

    key_df = df.drop_duplicates(subset=groups)

    for i in range(len(key_df)):
        tmp = summary
        current_filename = file_prefix
        for key in groups:
            tmp = tmp[tmp[key] == key_df[key].iloc[i]]
            current_filename += f"{filename_joiner if len(current_filename) > 0 and not current_filename.endswith('/') else ''}{key_df[key].iloc[i]}"
        tmp.to_csv(f"{current_filename}.txt", sep="\t", index=False)


def box_plot(df: DataFrame, x: str, y: str, groups: Union[List[str], str] = None, file_prefix: str = "",
             filename_joiner: str = "_"):
    if isinstance(groups, str):
        groups = [groups]
    if groups is None or len(groups) == 0:
        _box_plot(df, x, y, file_prefix)

    else:
        key_df = df.drop_duplicates(subset=groups)

        for i in range(len(key_df)):
            tmp = df
            current_filename = file_prefix
            for key in groups:
                tmp = tmp[tmp[key] == key_df[key].iloc[i]]
                current_filename += f"{filename_joiner if len(current_filename) > 0 else ''}{key_df[key].iloc[i]}"
            _box_plot(tmp, x, y, current_filename)


def _box_plot(df: DataFrame, x: str, y: str, file_name: str):
    plt.figure(visible=False)
    data = []
    for xi in df[x].unique():
        data.append([k for k in df[df[x] == xi][y] if str(k) != "nan"])

    bp = plt.boxplot(data, showmeans=False)

    minimums = [round(item.get_ydata()[0], 1) for item in bp['caps']][::2]
    q1 = [round(min(item.get_ydata()), 1) for item in bp['boxes']]
    medians = [item.get_ydata()[0] for item in bp['medians']]
    q3 = [round(max(item.get_ydata()), 1) for item in bp['boxes']]
    maximums = [round(item.get_ydata()[0], 1) for item in bp['caps']][1::2]

    rows = [df[x].unique().tolist(), minimums, q1, medians, q3, maximums]

    with open(f"{file_name}.txt", "w") as bp_file:
        for row in rows:
            bp_file.write("\t".join(map(str, row)) + "\n")


if __name__ == '__main__':
    target_folder = "../pgfplots"
    environments = ["pointmaze", "robotmaze", "hopper_uni", "walker2d_uni"]

    dfs = []
    validation_dfs = []
    for seed in range(30):
        for sampling in ["both", "s1", "s2", "mapelites", "ga", "ne"]:
            for env in environments:
                sampling_string = sampling \
                    if (sampling.startswith("map") or sampling.startswith("ga")) else "bimapelites_" + sampling
                if sampling == "ne":
                    sampling_string = "ne"
                else:
                    sampling_string += "_cgp"
                run = f"{sampling_string}_{env}_{seed}"

                df = pd.read_csv(f"../results/{run}.csv")
                df["seed"] = seed
                df["sampling"] = sampling
                df["environment"] = env
                if sampling == "ne":
                    df["coverage1"] = df["coverage2"] = df["coverage"]
                dfs.append(df)

                validation_df = pd.read_csv(f"../results/{run}_validation.csv")
                validation_df["seed"] = seed
                validation_df["sampling"] = sampling
                validation_df["environment"] = env
                validation_dfs.append(validation_df)

    df = pd.concat(dfs, ignore_index=True)
    validation_df = pd.concat(validation_dfs, ignore_index=True)
    df = df.dropna(subset=["max_fitness"])
    final_df = df[df["iteration"] == max(df["iteration"])]

    line_plot(df, x="iteration", y="max_fitness", groups=["environment", "sampling"],
              file_prefix=f"{target_folder}/fitness")
    line_plot(df[df["sampling"] != "ga"], x="iteration", y=["coverage1", "coverage2"],
              groups=["environment", "sampling"], file_prefix=f"{target_folder}/coverage")
    box_plot(final_df, x="sampling", y="max_fitness", groups=["environment"],
             file_prefix=f"{target_folder}/final_fitness")
    filtered_df = validation_df[(validation_df["repertoire_id"] == 1) | (validation_df["repertoire_id"] == "main")]
    box_plot(filtered_df, x="sampling", y="max_validation_fitness", groups=["environment"],
             file_prefix=f"{target_folder}/validation_fitness")
