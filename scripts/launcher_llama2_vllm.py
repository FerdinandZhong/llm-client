from llm_client.handler import start_vllm_server
from llm_client.utils import read_yaml
from llm_client.schemas import VllmServerConfig
import os
import signal

if __name__ == "__main__":
    config_yaml = "/root/Projects/llm-client/config_yamls/llama2-13b-chat-vllm.yaml"
    vllm_server_config = VllmServerConfig.parse_obj(read_yaml(config_yaml)["server_config"])
    proc = start_vllm_server(vllm_server_config, "/root/experiments_logs/llama2-13b.txt")
    try:
        proc.wait()
    except KeyboardInterrupt:
        try:
            proc.terminate()
            print("proc terminated")
        except OSError:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait()
    