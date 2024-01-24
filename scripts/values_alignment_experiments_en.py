import pandas as pd
from llm_client.pipeline import Pipeline
import time
from llm_client.values_alignment.experiments_utils import get_experiment_result
from llm_client.values_alignment.experiments_utils import generate_question_prompts
import argparse
import asyncio


parser = argparse.ArgumentParser(description='Llama model launcher')

# Add arguments
parser.add_argument('--config_yaml_name', type=str, default="llama2-7b-chat-vllm",
                help='llama2 model config yaml name')

parser.add_argument('--output_model_path', type=str, default="Llama2_7b")
parser.add_argument('--shuffle', action='store_true',
                   help='whether do shuffle')
parser.add_argument('--chunk_size', type=int, default=50)
parser.add_argument('--only_appending', action='store_true')

args = parser.parse_args()

root_path = "/root/Projects/llm-client/notebooks/values_alignment"

en_additional_prompt = "Response in JSON format: {\"answer_number\": 1, \"reason\": \"this is the reason\"}\n"

en_question_df = pd.read_csv(f"{root_path}/vsm2013_english_questions.csv")
en_question_df = en_question_df.where(pd.notnull(en_question_df), None)


if __name__ == "__main__":
    prompt_list = generate_question_prompts(question_df=en_question_df)
    en_context_df = pd.read_csv(f"{root_path}/vsm_english_context.csv")

    config_yaml = f"/root/Projects/llm-client/config_yamls/{args.config_yaml_name}.yaml"
    pipeline = Pipeline(config_yaml, verbose=1)

    shuffle_path = "no_shuffle" if not args.shuffle else "shuffle"
    output_path = root_path + f"/experiments_results/{args.output_model_path}/english/{shuffle_path}" +"/result_{seed}.csv"

    asyncio.run(get_experiment_result(
        question_prompts = prompt_list,
        experiment_context = en_context_df,
        output_path = output_path,
        pipeline = pipeline,
        chunk_size = args.chunk_size,
        use_random_options = args.shuffle,
        additional_prompt = en_additional_prompt,
        only_for_appending = args.only_appending
    ))