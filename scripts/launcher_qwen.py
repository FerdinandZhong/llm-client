from llm_client.handler import start_vllm_server
from llm_client.utils import read_yaml
from llm_client.schemas import VllmServerConfig
import os
import signal
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Qwen model launcher')

    # Add arguments
    parser.add_argument('--config_yaml_name', type=str, default="qwen-14b-chat-vllm",
                    help='qwen model config yaml name')

    parser.add_argument('--log_file', type=str, default="qwen-14b")

    # Parse the arguments
    args = parser.parse_args()

    config_yaml = f"/root/Projects/llm-client/config_yamls/{args.config_yaml_name}.yaml"
    vllm_server_config = VllmServerConfig.parse_obj(read_yaml(config_yaml)["server_config"])
    proc = start_vllm_server(vllm_server_config, f"/root/experiments_logs/{args.log_file}.txt")
    try:
        proc.wait()
    except KeyboardInterrupt:
        try:
            proc.terminate()
            print("proc terminated")
        except OSError:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait()
    