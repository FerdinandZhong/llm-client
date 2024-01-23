import asyncio
import random
import re
import json
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

from ..pipeline import Pipeline
import scipy.stats as stats
import numpy as np
import math


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
        question_prompts.append(
            (prompt.format(question=row["questions"]), option_list)
        )

    return question_prompts


async def get_experiment_result(
    question_prompts: List[Tuple[str, List[str]]],
    experiment_context: pd.DataFrame,
    output_path: str,
    pipeline: Pipeline,
    chunk_size: int = 50,
    use_random_options: bool = False,
    additional_prompt: str = "",
):
    for seed in (seed_pbar := tqdm(range(10))):
        seed_pbar.set_description(f"seed: {seed}")
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
                    response_list = await asyncio.gather(
                        *[
                            pipeline.model_predict(full_prompt)
                            for full_prompt in chunk_list
                        ]
                    )

                    for question_response, response_context_index, chunk_index in zip(
                        response_list, context_index_list, question_index_list
                    ):
                        current_df.loc[response_context_index, f"m_{chunk_index + 1}"] = (
                            question_response
                        )

                    chunk_list = []
                    context_index_list = []
                    question_index_list = []
                else:
                    continue
            
        if len(chunk_list) > 0:
            response_list = await asyncio.gather(
                *[
                    pipeline.model_predict(full_prompt)
                    for full_prompt in chunk_list
                ]
            )

            for question_response, chunk_index in zip(
                response_list, question_index_list
            ):
                current_df.loc[context_index, f"m_{chunk_index + 1}"] = (
                    question_response
                )

        current_df.to_csv(output_path.format(seed=seed))


def retrieve_score(all_seeds_result_dfs, *context_cols):
    score_df_list = []

    pattern = (
        r"(?<answer_number\": )\d|(?<answer\\_number\": )\d|(?<answer_number\": \")\d"
        r"(?<answer_number\":)\d|(?<answer\\_number\":)\d|(?<answer_number\":\")\d"
        r"(?<answer\": )\d|(?<answer\": \")\d|(?<answer\":)\d|(?<answer\":\")\d"
    )

    for seed_result_df in all_seeds_result_dfs:
        seed_score_df = pd.DataFrame(columns=context_cols)
        for index, row in seed_result_df.iterrows():
            seed_score_df.loc[index, context_cols] = [
                row[context_col] for context_col in context_cols
            ]
            for question_index in range(1, 25):
                question_col = f"m_{question_index}"
                try:
                    json_result = json.loads(row[question_col].lstrip().rstrip())
                    seed_score_df.loc[index, question_col] = float(
                        json_result["answer_number"]
                    )
                except KeyError:
                    try:
                        seed_score_df.loc[index, question_col] = float(
                            json_result["answer"]
                        )
                    except KeyError:
                        seed_score_df.loc[index, question_col] = float(
                            json_result[list(json_result.keys())[0]]
                        )
                except Exception:
                    result = re.search(pattern, row[question_col])
                    if result:
                        seed_score_df.loc[index, question_col] = float(result.group())
                    else:
                        print(f"No match found. {row[question_col]}")
                        seed_score_df.loc[index, question_col] = 3

        score_df_list.append(seed_score_df)

    print(score_df_list[0].head())
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
    score_df_list, const: int = 0, keep_m15: bool = True, keep_m18: bool = True
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
    grouped_avg_df = source_df.groupby(merge_by)[target_cols].agg(["mean"])

    grouped_avg_df.columns = grouped_avg_df.columns.get_level_values(0)
    grouped_avg_df.reset_index(inplace=True)

    transposed_grouped_avg_df = grouped_avg_df.T
    score_corr_matrix = transposed_grouped_avg_df.corr(method="pearson")
    upper = score_corr_matrix.where(np.triu(np.ones(score_corr_matrix.shape), k=1).astype(bool))

    p_values = []
    for col1 in upper.columns:
        for col2 in upper.columns:
            if col1 == col2:
                continue
            corr_score, p_val = stats.pearsonr(transposed_grouped_avg_df[col1], transposed_grouped_avg_df[col2])
            p_values.append({"col1": col1, "col2": col2, "corr_score": corr_score, "p_value": p_val})
    
    p_values_df = pd.DataFrame(p_values)

    overall_pearson_correlation_score = p_values_df["corr_score"].mean()
    overall_pearson_p_values = p_values_df["p_value"].mean()

    return p_values_df, overall_pearson_correlation_score, overall_pearson_p_values

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

    std_mappings = {}
    for col in target_cols:
        std_mappings[col] = grouped_avg_df[col].std()

    return std_mappings


def compute_distances_two_dfs(source_df_1, source_df_2, target_cols, merge_by):
    grouped_avg_df_1 = source_df_1.groupby(merge_by)[target_cols].agg(["mean"])
    grouped_avg_df_2 = source_df_2.groupby(merge_by)[target_cols].agg(["mean"])
    correlation_df = source_df_1.loc[:, [merge_by]].copy()

    for row_index in range(grouped_avg_df_1.shape[0]):
        series_1 = grouped_avg_df_1.loc[row_index][target_cols].to_numpy()
        series_2 = grouped_avg_df_2.loc[row_index][target_cols].to_numpy()

        corr_score, p_value = stats.pearsonr(series_1, series_2)

        correlation_df.loc["row_index", "corr_score"] = corr_score
        correlation_df.loc["row_index", "p_value"] = p_value
    
    return correlation_df

def compute_vsm_values_gap(source_df_1, source_df_2, target_cols, merge_by):
    grouped_avg_df_1 = source_df_1.groupby(merge_by)[target_cols].agg(["mean"])
    grouped_avg_df_2 = source_df_2.groupby(merge_by)[target_cols].agg(["mean"])
    distance_df = source_df_1.loc[:, [merge_by]].copy()

    for row_index in range(grouped_avg_df_1.shape[0]):
        for target_col in target_cols:
            value_1 = grouped_avg_df_1.loc[row_index, target_col]
            value_2 = grouped_avg_df_2.loc[row_index, target_col]
            distance_df.loc[row_index, target_col] = math.sqrt(math.pow((value_1 - value_2), 2))

    overall_distance = distance_df[target_cols].mean()

    return distance_df, overall_distance


