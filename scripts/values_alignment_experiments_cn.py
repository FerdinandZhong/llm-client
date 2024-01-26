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
parser.add_argument('--chunk_size', type=int, default=50)
parser.add_argument('--cn_use_en', action='store_true')

args = parser.parse_args()

root_path = "/root/Projects/llm-client/notebooks/values_alignment"

cn_additional_prompt = "以此JSON格式回答：{\"答案序号\": 1, \"原因\": \"中文回答的理由\"}\n"
en_additional_prompt = "Response in JSON format: {\"answer_number\": 1, \"reason\": \"this is the reason\"}\n"

cn_question_df = pd.read_csv(f"{root_path}/vsm2013_chinese_questions.csv")
cn_question_df = cn_question_df.where(pd.notnull(cn_question_df), None)


if __name__ == "__main__":
    prompt_list = generate_question_prompts(question_df=cn_question_df)
    print(prompt_list)
    cn_context_df = pd.read_csv(f"{root_path}/vsm_chinese_context.csv")

    config_yaml = f"/root/Projects/llm-client/config_yamls/{args.config_yaml_name}.yaml"
    pipeline = Pipeline(config_yaml, verbose=1)

    if not args.cn_use_en:
        cn_output_path = root_path + f"/experiments_results/{args.output_model_path}/" + "chinese/result_{seed}.csv"
    else:
        cn_output_path = root_path + f"/experiments_results/{args.output_model_path}/" + "chinese_en/result_{seed}.csv"

    asyncio.run(get_experiment_result(
        question_prompts = prompt_list,
        experiment_context = cn_context_df,
        output_path = cn_output_path,
        pipeline = pipeline,
        chunk_size = args.chunk_size,
        use_random_options = False,
        additional_prompt = cn_additional_prompt if not args.cn_use_en else en_additional_prompt,
        # customized_range = range(1, 10)
    ))