import asyncio
import json
import math
import random
import re
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from tqdm import tqdm

from ..pipeline import Pipeline
from .constants import country_mapping


def generate_question_prompts(question_df):
    """
    Generate question prompts from a DataFrame.

    Each row in the DataFrame is expected to contain a 'prompt' field and five 'option' fields.
    The 'prompt' field is a string that can contain placeholders for variables, which are replaced with values from the DataFrame.
    The 'option' fields are strings that represent possible answers to the question.

    Args:
        question_df (pandas.DataFrame): A DataFrame where each row represents a question.

    Returns:
        list: A list of tuples. Each tuple contains a formatted question prompt and a list of options.
    """
    question_prompts = []
    for row in question_df.itertuples(False):
        row = row._asdict()
        if row["prompt"] is not None:
            prompt = str(row["prompt"])
            option_list = []
            for i in range(5):
                option_list.append(row[f"option_{i+1}"])
        question_prompts.append((prompt.format(question=row["questions"]), option_list))

    return question_prompts


async def get_experiment_result(
    question_prompts: List[Tuple[str, List[str]]],
    experiment_context: pd.DataFrame,
    output_path: str,
    pipeline: Pipeline,
    chunk_size: int = 50,
    use_random_options: bool = False,
    additional_prompt: str = "",
    only_for_appending: bool = False,
    customized_range: List[int] = None,
):
    loop_range = range(10) if customized_range is None else customized_range
    for seed in (seed_pbar := tqdm(loop_range)):
        seed_pbar.set_description(f"seed: {seed}")
        if only_for_appending:
            current_df = pd.read_csv(output_path.format(seed=seed), index_col=0)
        else:
            current_df = experiment_context.copy()

        chunk_list = []
        context_index_list = []
        question_index_list = []
        for context_index, row in (
            context_pbar := tqdm(list(current_df.iterrows()), total=current_df.shape[0])
        ):
            context_pbar.set_description(f"context: {context_index}")
            gender = row["gender"]
            age = row["age"]
            nation = row["nation"]
            city = row["city"]

            # full_list_len = len(question_prompts)

            for question_index, (question_prompt, option_list) in enumerate(
                question_prompts
            ):
                if use_random_options:
                    random.seed(seed)
                    shuffled_options = random.sample(option_list, len(option_list))
                    str_options = "\n".join(shuffled_options)
                else:
                    str_options = "\n".join(option_list)
                full_prompt = additional_prompt + question_prompt.format(
                    options=str_options,
                    gender=gender,
                    age=age,
                    nation=nation,
                    city=city,
                )
                chunk_list.append(full_prompt)
                context_index_list.append(context_index)
                question_index_list.append(question_index)

                if len(chunk_list) >= chunk_size:
                    if only_for_appending:
                        pass
                    else:
                        response_list = await asyncio.gather(
                            *[
                                pipeline.model_predict(full_prompt)
                                for full_prompt in chunk_list
                            ]
                        )

                        for (
                            question_response,
                            response_context_index,
                            chunk_index,
                        ) in zip(
                            response_list, context_index_list, question_index_list
                        ):
                            current_df.loc[
                                response_context_index, f"m_{chunk_index + 1}"
                            ] = question_response

                    chunk_list = []
                    context_index_list = []
                    question_index_list = []
                else:
                    continue

        if len(chunk_list) > 0:
            response_list = await asyncio.gather(
                *[pipeline.model_predict(full_prompt) for full_prompt in chunk_list]
            )

            for question_response, response_context_index, chunk_index in zip(
                response_list, context_index_list, question_index_list
            ):
                current_df.loc[response_context_index, f"m_{chunk_index + 1}"] = (
                    question_response
                )

        current_df.to_csv(output_path.format(seed=seed))


def retrieve_score(all_seeds_result_dfs, *context_cols):
    total_num_response = 0
    not_matched_result = 0
    score_df_list = []

    pattern = (
        r"(?<=[aA]nswer[_ ][nN]umber[\"']: )\d|(?<=answer\\_[nN]umber[\"']: )\d|(?<=[aA]nswer[_ ][nN]umber[\"']: [\"'])\d|"
        r"(?<=[aA]nswer[_ ][nN]umber[\"']:)\d|(?<=answer\\_[nN]umber[\"']:)\d|(?<=[aA]nswer[_ ][nN]umber[\"']:[\"'])\d|"
        r"(?<=[aA]nswer[\"']: )\d|(?<=[aA]nswer[\"']: [\"'])\d|(?<=[aA]nswer[\"']:)\d|(?<=[aA]nswer[\"']:[\"'])\d|"
        r"(?<=[aA]nswer[_ ][nN]umber: )\d|(?<=answer\\_[nN]umber: )\d|(?<=[aA]nswer[_ ][nN]umber: [\"'])\d|"
        r"(?<=[aA]nswer[_ ][nN]umber:)\d|(?<=answer\\_[nN]umber:)\d|(?<=[aA]nswer[_ ][nN]umber:[\"'])\d|"
        r"(?<=[aA]nswer: )\d|(?<=[aA]nswer: [\"'])\d|(?<=[aA]nswer:)\d|(?<=[aA]nswer:[\"'])\d|"
        r"(?<=答案序号[\"']: )\d|(?<=答案序号[\"']: [\"'])\d|"
        r"(?<=答案序号[\"']:)\d|(?<=答案序号[\"']:[\"'])\d"
    )

    for seed, seed_result_df in enumerate(all_seeds_result_dfs):
        seed_score_df = pd.DataFrame(columns=context_cols)
        for index, row in seed_result_df.iterrows():
            seed_score_df.loc[index, context_cols] = [
                row[context_col] for context_col in context_cols
            ]
            for question_index in range(1, 25):
                total_num_response += 1
                question_col = f"m_{question_index}"
                try:
                    json_result = json.loads(str(row[question_col]).lstrip().rstrip())
                    seed_score_df.loc[index, question_col] = float(
                        json_result[list(json_result.keys())[0]]
                    )

                except (
                    json.JSONDecodeError,
                    TypeError,
                    AttributeError,
                    ValueError,
                    IndexError,
                ):
                    try:
                        result = re.search(pattern, row[question_col])
                        if result:
                            seed_score_df.loc[index, question_col] = float(
                                result.group()
                            )
                        else:
                            print(f"No match found. {row[question_col]}")
                            not_matched_result += 1
                            seed_score_df.loc[index, question_col] = 3
                    except TypeError:
                        print(
                            f"type error: {seed}, {index}, {question_index}, {row[question_col]}"
                        )
                        seed_score_df.loc[index, question_col] = 3

        score_df_list.append(seed_score_df)

    print(score_df_list[0].head())
    print(
        f"total number of not matched: {not_matched_result}, percentage: {not_matched_result/total_num_response}"
    )
    return score_df_list


def get_avg_score_df(score_df_list):
    """
    Calculate the average score dataframe from a list of score dataframes.

    Parameters:
    score_df_list (List[pd.DataFrame]): A list of pandas dataframes containing scores.

    Returns:
    pd.DataFrame: A dataframe containing the average scores.
    """
    avg_score_df = score_df_list[0].copy()
    question_cols = [f"m_{question_index}" for question_index in range(1, 25)]

    for seed_score_df in score_df_list[1:]:
        for index, row in seed_score_df.iterrows():
            for question_col in question_cols:
                avg_score_df.loc[index, question_col] += int(row[question_col])

    for col in question_cols:
        avg_score_df[col] = avg_score_df[col].apply(lambda x: x / len(score_df_list))

    return avg_score_df


def get_vsm_score_df(
    score_df_list, const: int = 0, keep_m15: bool = False, keep_m18: bool = False
):
    """
    Calculate the VSM score dataframe from a list of score dataframes.

    Parameters:
    score_df_list (List[pd.DataFrame]): A list of pandas dataframes containing scores.
    const (int): A constant value to be added to the score calculation. Default is 0.
    keep_m15 (bool): Whether to keep the score for m15. Default is True.
    keep_m18 (bool): Whether to keep the score for m18. Default is True.

    Returns:
    pd.DataFrame: A dataframe containing the VSM scores.
    """
    # rest of the function here
    vsm_score_df = score_df_list[0].loc[:, ["gender", "age", "nation", "city"]].copy()
    columns = ["PDI", "IDV", "MAS", "UAI", "LTO", "IVR"]

    for col in columns:
        vsm_score_df[col] = 0

    for _, score_df in enumerate(score_df_list):
        for index, row in score_df.iterrows():
            PDI_score = (
                35 * (int(row["m_7"]) - int(row["m_2"]))
                + 25 * (int(row["m_20"]) - int(row["m_23"]))
                + const
            )
            vsm_score_df.loc[index, "PDI"] += PDI_score

            IDV_score = (
                35 * (int(row["m_4"]) - int(row["m_1"]))
                + 35 * (int(row["m_9"]) - int(row["m_6"]))
                + const
            )
            vsm_score_df.loc[index, "IDV"] += IDV_score

            MAS_score = (
                35 * (int(row["m_5"]) - int(row["m_3"]))
                + 35 * (int(row["m_8"]) - int(row["m_10"]))
                + const
            )
            vsm_score_df.loc[index, "MAS"] += MAS_score

            m15 = int(row["m_15"]) if keep_m15 else 3
            m18 = int(row["m_18"]) if keep_m18 else 3
            UAI_score = (
                40 * (m18 - m15) + 25 * (int(row["m_21"]) - int(row["m_24"])) + const
            )
            vsm_score_df.loc[index, "UAI"] += UAI_score

            LTO_score = (
                40 * (int(row["m_13"]) - int(row["m_14"]))
                + 25 * (int(row["m_19"]) - int(row["m_22"]))
                + const
            )
            vsm_score_df.loc[index, "LTO"] += LTO_score

            IVR_score = (
                35 * (int(row["m_12"]) - int(row["m_11"]))
                + 40 * (int(row["m_17"]) - int(row["m_16"]))
                + const
            )
            vsm_score_df.loc[index, "IVR"] += IVR_score

    for col in columns:
        vsm_score_df[col] = vsm_score_df[col].apply(lambda x: x / len(score_df_list))

    return vsm_score_df


def get_original_score_pearson_values(source_df, target_cols, merge_by):
    """
    Compute Pearson correlation values between different columns of a DataFrame.

    Parameters:
    source_df (pd.DataFrame): The source DataFrame.
    target_cols (List[str]): The target columns for which to compute correlations.
    merge_by (str): The column to group by before computing correlations.

    Returns:
    tuple: A tuple containing a DataFrame of Pearson correlation values, the mean correlation score, and the mean p-value.
    """
    # rest of the function here
    grouped_avg_df = source_df.groupby(merge_by)[target_cols].agg(["mean"])

    grouped_avg_df.columns = grouped_avg_df.columns.get_level_values(0)
    grouped_avg_df.reset_index(inplace=True)

    transposed_grouped_avg_df = grouped_avg_df.T
    transposed_grouped_avg_df.columns = transposed_grouped_avg_df.iloc[0]
    transposed_grouped_avg_df = transposed_grouped_avg_df[1:]

    p_values = []
    for col1 in transposed_grouped_avg_df.columns:
        for col2 in transposed_grouped_avg_df.columns:
            if col1 == col2:
                continue
            corr_score, p_val = stats.pearsonr(
                transposed_grouped_avg_df[col1], transposed_grouped_avg_df[col2]
            )
            p_values.append(
                {"col1": col1, "col2": col2, "corr_score": corr_score, "p_value": p_val}
            )

    p_values_df = pd.DataFrame(p_values)

    overall_pearson_correlation_score = p_values_df["corr_score"].mean()
    overall_pearson_p_values = p_values_df["p_value"].mean()

    return p_values_df, overall_pearson_correlation_score, overall_pearson_p_values

def get_overall_pearson_correlation_values(source_df, target_df):
    source_df.loc[:,"m_15"] = 3
    source_df.loc[:,"m_18"] = 3
    target_df.loc[:,"m_15"] = 3
    target_df.loc[:,"m_18"] = 3
    source_vector = source_df.mean()
    target_vector = target_df.mean()

    corr_score, p_val = stats.pearsonr(
        source_vector, target_vector
    )
    return corr_score, p_val

def get_standard_varaince_over_df(source_df, target_cols, merge_by):
    """
    Calculate the standard variance over a dataframe.

    Parameters:
    source_df (pd.DataFrame): The source dataframe.
    target_cols (List[str]): The target columns.
    merge_by (str): The column to merge by.

    Returns:
    dict: A dictionary mapping each target column to its standard deviation.
    """
    grouped_avg_df = source_df.groupby(merge_by)[target_cols].agg(["mean"])
    grouped_avg_df.columns = grouped_avg_df.columns.get_level_values(0)
    std_all = 0

    std_mappings = {}
    for col in target_cols:
        std_mappings[col] = float(grouped_avg_df[col].std())
        std_all += float(grouped_avg_df[col].std())

    print(f"overall std: {std_all/6}")
    return std_mappings


def compute_distances_two_dfs(source_df_1, source_df_2, target_cols, merge_by):
    grouped_avg_df_1 = source_df_1.groupby(merge_by)[target_cols].agg(["mean"])
    grouped_avg_df_1.columns = grouped_avg_df_1.columns.get_level_values(0)
    grouped_avg_df_1.reset_index(inplace=True)
    grouped_avg_df_2 = source_df_2.groupby(merge_by)[target_cols].agg(["mean"])
    grouped_avg_df_2.columns = grouped_avg_df_2.columns.get_level_values(0)
    correlation_df = pd.DataFrame(index=grouped_avg_df_1.index)

    for row_index in range(grouped_avg_df_1.shape[0]):
        series_1 = grouped_avg_df_1.iloc[row_index][target_cols].to_numpy()
        series_2 = grouped_avg_df_2.iloc[row_index][target_cols].to_numpy()

        corr_score, p_value = stats.pearsonr(series_1, series_2)

        correlation_df.loc[row_index, "corr_score"] = corr_score
        correlation_df.loc[row_index, "p_value"] = p_value
    return correlation_df


def compute_vsm_values_gap(source_df_1, source_df_2, target_cols, merge_by):
    """
    Compute the gap between VSM values of two dataframes.

    Parameters:
    source_df_1 (pd.DataFrame): The first source dataframe.
    source_df_2 (pd.DataFrame): The second source dataframe.
    target_cols (List[str]): The target columns.
    merge_by (str): The column to merge by.

    Returns:
    pd.DataFrame: A dataframe containing the gap between VSM values.
    """
    grouped_avg_df_1 = source_df_1.groupby(merge_by)[target_cols].agg(["mean"])
    grouped_avg_df_1.columns = grouped_avg_df_1.columns.get_level_values(0)
    grouped_avg_df_2 = source_df_2.groupby(merge_by)[target_cols].agg(["mean"])
    grouped_avg_df_2.columns = grouped_avg_df_2.columns.get_level_values(0)
    distance_df = pd.DataFrame(columns=target_cols)

    for row_index in range(grouped_avg_df_1.shape[0]):
        for target_col in target_cols:
            value_1 = grouped_avg_df_1.iloc[row_index][target_col]
            value_2 = grouped_avg_df_2.iloc[row_index][target_col]
            distance_df.loc[row_index, target_col] = math.sqrt(
                math.pow((value_1 - value_2), 2)
            )

    overall_distance = distance_df[target_cols].mean()

    return distance_df, overall_distance


def compute_vsm_values_center_gap(source_df_1, source_df_2, target_cols):

    grouped_avg_df_1 = source_df_1[target_cols].mean()
    grouped_avg_df_2 = source_df_2[target_cols].mean()
    print(f"df1 centroid: \n{grouped_avg_df_1}")
    print(f"df2 centroid: \n{grouped_avg_df_2}")
    distance_mapping = {}
    total_distance = 0

    for target_col in target_cols:
        value_1 = grouped_avg_df_1[target_col]
        value_2 = grouped_avg_df_2[target_col]
        distance = math.sqrt(math.pow((value_1 - value_2), 2))
        distance_mapping[target_col] = distance
        total_distance += distance

    print(f"overall centroids distance: {total_distance/6}")
    return distance_mapping


def visualize_vsm_differences(
    df_1: pd.DataFrame,
    df_2: pd.DataFrame,
    merge_by: str,
    target_cols: List[str],
    label_1: str,
    label_2: str,
    savepath: str,
    color_1: str = "green",
    color_2: str = "blue",
    figsize: Tuple = (15, 10),
):
    grouped_avg_df_1 = df_1.groupby(merge_by)[target_cols].agg(["mean"])
    grouped_avg_df_1.columns = grouped_avg_df_1.columns.get_level_values(0)
    grouped_avg_df_2 = df_2.groupby(merge_by)[target_cols].agg(["mean"])
    grouped_avg_df_2.columns = grouped_avg_df_2.columns.get_level_values(0)
    grouped_avg_df_1.reset_index(inplace=True)
    grouped_avg_df_2.reset_index(inplace=True)

    _, axes = plt.subplots(1, len(target_cols), figsize=figsize)
    handles = []
    labels = []

    for i, column in enumerate(target_cols):
        for x, (y_1, y_2) in enumerate(
            zip(grouped_avg_df_1[column], grouped_avg_df_2[column])
        ):
            scatter_1 = axes[i].scatter(x, y_1, color=color_1, label=label_1)
            axes[i].text(
                x=x,
                y=y_1 + 1.0,
                s=country_mapping[grouped_avg_df_1.iloc[x][merge_by]],
                ha="center",
            )
            scatter_2 = axes[i].scatter(x, y_2, color=color_2, label=label_2)
            axes[i].text(
                x=x,
                y=y_2 + 1.0,
                s=country_mapping[grouped_avg_df_1.iloc[x][merge_by]],
                ha="center",
            )  # use same name
            axes[i].set_ylim([-100, 100])

        axes[i].set_title(column)
        # Collect handles and labels for the legend
        if i == 0:
            handles.extend([scatter_1, scatter_2])
            labels.extend([label_1, label_2])

    plt.tight_layout()
    plt.legend(handles, labels, loc="upper right")
    plt.savefig(savepath)
    plt.show()


def visualize_more_vsm_differences(
    df_1: pd.DataFrame,
    df_2: pd.DataFrame,
    df_3: pd.DataFrame,
    merge_by: str,
    target_cols: List[str],
    label_1: str,
    label_2: str,
    label_3: str,
    savepath: str,
    color_1: str = "green",
    color_2: str = "blue",
    color_3: str = "red",
    figsize: Tuple = (15, 10),
):
    grouped_avg_df_1 = df_1.groupby(merge_by)[target_cols].agg(["mean"])
    grouped_avg_df_1.columns = grouped_avg_df_1.columns.get_level_values(0)
    grouped_avg_df_2 = df_2.groupby(merge_by)[target_cols].agg(["mean"])
    grouped_avg_df_2.columns = grouped_avg_df_2.columns.get_level_values(0)
    grouped_avg_df_3 = df_3.groupby(merge_by)[target_cols].agg(["mean"])
    grouped_avg_df_3.columns = grouped_avg_df_2.columns.get_level_values(0)
    grouped_avg_df_1.reset_index(inplace=True)
    grouped_avg_df_2.reset_index(inplace=True)
    grouped_avg_df_3.reset_index(inplace=True)

    _, axes = plt.subplots(1, len(target_cols), figsize=figsize)
    handles = []
    labels = []

    for i, column in enumerate(target_cols):
        for x, (y_1, y_2, y_3) in enumerate(
            zip(
                grouped_avg_df_1[column],
                grouped_avg_df_2[column],
                grouped_avg_df_3[column],
            )
        ):
            scatter_1 = axes[i].scatter(x, y_1, color=color_1, label=label_1)
            axes[i].text(
                x=x,
                y=y_1 + 1.0,
                s=country_mapping[grouped_avg_df_1.iloc[x][merge_by]],
                ha="center",
            )
            scatter_2 = axes[i].scatter(x, y_2, color=color_2, label=label_2)
            axes[i].text(
                x=x,
                y=y_2 + 1.0,
                s=country_mapping[grouped_avg_df_1.iloc[x][merge_by]],
                ha="center",
            )  # use same name
            scatter_3 = axes[i].scatter(x, y_3, color=color_3, label=label_3)
            axes[i].text(
                x=x,
                y=y_3 + 1.0,
                s=country_mapping[grouped_avg_df_1.iloc[x][merge_by]],
                ha="center",
            )  # use same name
            axes[i].set_ylim([-100, 100])

        axes[i].set_title(column)
        # Collect handles and labels for the legend
        if i == 0:
            handles.extend([scatter_1, scatter_2, scatter_3])
            labels.extend([label_1, label_2, label_3])

    plt.tight_layout()
    plt.legend(handles, labels, loc="upper right")

    plt.savefig(savepath)
    plt.show()


def visualize_vsm_differences_in_box(
    df_1: pd.DataFrame,
    df_2: pd.DataFrame,
    target_cols: List[str],
    label_1: str,
    label_2: str,
    savepath: str,
    color_1: str = "lightgreen",
    color_2: str = "lightblue",
    figsize: Tuple = (15, 10),
    y_range: List = [-150, 150]
):
    # grouped_avg_df_1 = df_1.groupby(merge_by)[target_cols].agg(["mean"])
    # grouped_avg_df_1.columns = grouped_avg_df_1.columns.get_level_values(0)
    # grouped_avg_df_2 = df_2.groupby(merge_by)[target_cols].agg(["mean"])
    # grouped_avg_df_2.columns = grouped_avg_df_2.columns.get_level_values(0)
    # grouped_avg_df_1.reset_index(inplace=True)
    # grouped_avg_df_2.reset_index(inplace=True)

    _, axes = plt.subplots(1, len(target_cols), figsize=figsize)

    for i, column in enumerate(target_cols):
        # for y_1, y_2 in zip(grouped_avg_df_1[column], grouped_avg_df_2[column]):
        boxplot_1 = axes[i].boxplot(
            df_1[column],
            positions=[1],
            patch_artist=True,
            labels=[label_1],
            whiskerprops=dict(color="gray"),
            meanprops=dict(color="red"),
        )
        for patch in boxplot_1["boxes"]:
            patch.set_facecolor(color_1)
        boxplot_2 = axes[i].boxplot(
            df_2[column],
            positions=[2],
            patch_artist=True,
            labels=[label_2],
            whiskerprops=dict(color="gray"),
            meanprops=dict(color="red"),
        )
        for patch in boxplot_2["boxes"]:
            patch.set_facecolor(color_2)
        axes[i].set_ylim(y_range)

        axes[i].set_title(column)
        axes[i].grid(True, linestyle="--", alpha=0.7)
        # Collect handles and labels for the legend

    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()


def visualize_multiple_vsm_differences_in_box(
    df_list: List[pd.DataFrame],
    label_list: List[str],
    color_list: List[str],
    target_cols: List[str],
    savepath: str,
    figsize: Tuple = (15, 10),
):
    """
    Visualize the differences between multiple VSM (Value Set Model) scores in box plots.
    
    This function takes a list of dataframes, each containing VSM scores, and plots the differences
    between these scores for each target column. The plots are saved to the specified savepath.
    
    Args:
        df_list (List[pd.DataFrame]): A list of dataframes containing VSM scores.
        label_list (List[str]): A list of labels corresponding to each dataframe in df_list.
        color_list (List[str]): A list of colors corresponding to each dataframe in df_list.
        merge_by (str): The column to merge by before computing differences.
        target_cols (List[str]): The target columns for which to compute differences.
        savepath (str): The path where the plot will be saved.
        figsize (Tuple, optional): The size of the figure. Defaults to (15,  10).
    
    """
    assert len(df_list) == len(label_list) and len(df_list) == len(
        color_list
    ), "not equal length"
    # grouped_avg_df_list = []
    # for df in df_list:
    #     grouped_avg_df = df.groupby(merge_by)[target_cols].agg(["mean"])
    #     grouped_avg_df.columns = grouped_avg_df.columns.get_level_values(0)
    #     grouped_avg_df.reset_index(inplace=True)
    #     grouped_avg_df_list.append(grouped_avg_df)

    _, axes = plt.subplots(1, len(target_cols), figsize=figsize)

    for i, column in enumerate(target_cols):
        for index, df in enumerate(df_list):
            label = label_list[index]
            color = color_list[index]
            boxplot = axes[i].boxplot(
                df[column],
                positions=[index],
                patch_artist=True,
                labels=[label],
                whiskerprops={"color": "gray"},
                meanprops={"color": "red"},
            )
            for patch in boxplot["boxes"]:
                patch.set_facecolor(color)

        axes[i].set_ylim([-150, 150])

        axes[i].set_title(column)
        axes[i].grid(True, linestyle="--", alpha=0.7)
        # Collect handles and labels for the legend
            

    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()


def line_chart_of_dots(
        source_df, 
        target_df,
        label_1: str,
        label_2: str,
        savepath: str,
        color_1: str = "lightgreen",
        color_2: str = "lightblue",
        figsize: Tuple = (15, 10),
    ):
    source_vector = source_df.mean()
    target_vector = target_df.mean()

    plt.figure(figsize=figsize)  

    plt.plot(source_vector, 'o-', label=label_1, markersize=5, color=color_1)
    plt.plot(target_vector, 'o-', label=label_2, markersize=5, color=color_2)

    # Add labels and title
    plt.xlabel('Question Index')
    plt.ylabel('Score')
    plt.ylim([0, 5])

    # Add a legend
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()

def compute_vsm_values_center_distance(source_df_1, source_df_2, target_cols, standard_distance=20.33):

    grouped_avg_df_1 = source_df_1[target_cols].mean()
    grouped_avg_df_2 = source_df_2[target_cols].mean()
    total_distance = 0

    for target_col in target_cols:
        value_1 = grouped_avg_df_1[target_col]
        value_2 = grouped_avg_df_2[target_col]
        distance = math.sqrt(math.pow((value_1 - value_2), 2))
        total_distance += distance

    if standard_distance is not None:
        return round(total_distance/(6*standard_distance), 3)
    else:
        return round(total_distance/6, 3)

def generate_vsm_heatmap(source_df_mapping, target_df_mapping, savepath, vsm_cols, standard_distance=20.33, figsize=(15, 20)):
    columns = list(target_df_mapping.keys())

    vsm_distance_df = pd.DataFrame(columns=columns)

    for target_model, target_model_df in target_df_mapping.items():
        for source_model, source_model_df in source_df_mapping.items():
            vsm_distance_df.loc[source_model, target_model] = float(compute_vsm_values_center_distance(source_model_df, target_model_df, vsm_cols, standard_distance))
    
    vsm_distance_df = vsm_distance_df[vsm_distance_df.columns].astype(float)

    row_averages = vsm_distance_df.mean(axis=1)
    vsm_distance_df['Average'] = row_averages

    plt.figure(figsize=figsize)
    if standard_distance is not None:
        center = 45/standard_distance
    else:
        center=45
    sns.heatmap(vsm_distance_df, fmt=".3f", annot=True, cmap='Reds', center=1, xticklabels=True, yticklabels=True, linewidths=0.3)
    # for i, avg in enumerate(row_averages):
    #     plt.text(len(models_vsm_distance_df.columns), i, f"{avg:.2f}", ha='left', va='center', color='black')

    plt.tight_layout(pad=1.08)
    plt.xticks(rotation=45)
    plt.savefig(savepath, dpi=120)
    plt.show()


def generate_original_heatmap(source_df_mapping, target_df_mapping, savepath, question_cols, figsize=(15, 20)):
    columns = list(target_df_mapping.keys())

    score_distance_df = pd.DataFrame(columns=columns)

    for target_model, target_model_df in target_df_mapping.items():
        for source_model, source_model_df in source_df_mapping.items():
            score_distance_df.loc[source_model, target_model] = float(get_overall_pearson_correlation_values(source_model_df[question_cols], target_model_df[question_cols])[0])
    
    score_distance_df = score_distance_df[score_distance_df.columns].astype(float)

    row_averages = score_distance_df.mean(axis=1)
    score_distance_df['Average'] = row_averages

    plt.figure(figsize=figsize)
    sns.heatmap(score_distance_df, fmt=".3f", annot=True, cmap='Blues', center=1, xticklabels=True, yticklabels=True, linewidths=0.3)
    # for i, avg in enumerate(row_averages):
    #     plt.text(len(models_vsm_distance_df.columns), i, f"{avg:.2f}", ha='left', va='center', color='black')

    plt.tight_layout(pad=1.08)

    plt.savefig(savepath, dpi=120)
    plt.show()