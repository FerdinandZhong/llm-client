import asyncio
import random
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

from ..pipeline import Pipeline
import re
import json

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
    for seed in (seed_pbar := tqdm(range(1,10))):
        seed_pbar.set_description(f"seed: {seed}")
        current_df = experiment_context.copy()

        for context_index, row in (
            context_pbar := tqdm(current_df.iterrows(), total=current_df.shape[0])
        ):
            context_pbar.set_description(f"context: {context_index}")
            gender = row["gender"]
            age = row["age"]
            nation = row["nation"]
            city = row["city"]

            chunk_list = []
            chunk_index_list = []
            full_list_len = len(question_prompts)

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
                chunk_index_list.append(question_index)

                if len(chunk_list) >= chunk_size or question_index == full_list_len - 1:
                    response_list = await asyncio.gather(
                        *[
                            pipeline.model_predict(full_prompt)
                            for full_prompt in chunk_list
                        ]
                    )

                    for question_response, chunk_index in zip(
                        response_list, chunk_index_list
                    ):
                        current_df.loc[context_index, f"m_{chunk_index + 1}"] = (
                            question_response
                        )

                    chunk_list = []
                    chunk_index_list = []
                else:
                    continue

        current_df.to_csv(output_path.format(seed=seed))


def retrieve_score(all_seeds_result_dfs, *context_cols):
    score_df_list = []

    pattern = (
        r"(?<answer_number\": )\d|(?<answer\\_number\": )\d|(?<answer_number\": \")\d"
        r"(?<answer_number\":)\d|(?<answer\\_number\":)\d|(?<answer_number\":\")\d"
        r"(?<answer\": )\d|(?<answer\": \")\d|(?<answer\":)\d|(?<answer\":\")\d"
    )

    for seed_result_df in all_seeds_result_dfs:
        seed_score_df = pd.DataFrame(columns = context_cols)
        for index, row in seed_result_df.iterrows():
            seed_score_df.loc[index, context_cols] = [row[context_col] for context_col in context_cols]
            for question_index in range(1, 25):
                question_col = f"m_{question_index}"
                try:
                    json_result = josn.loads(row[question_col].lstrip().rstrip())
                    seed_score_df.loc[index, question_col] = float(json_result["answer_number"])
                except KeyError:
                    seed_score_df.loc[index, question_col] = float(json_result["answer"])
                except KeyError:
                    seed_score_df.loc[index, question_col] = float(json_result[list(json_result.keys())[0]])
                except Exception as e:
                    result = re.search(pattern, row[question_col])
                    if result:
                        seed_score_df.loc[index, question_col] = float(result.group())
                    else:
                        print(f"No match found. {row[question_col]}")
                        seed_score_df.loc[index, question_col] = 3
        
        score_df_list.append(seed_score_df)

    print(score_df_list[0].head())
    return score_df_list
        