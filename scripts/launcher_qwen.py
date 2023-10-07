from tgi_client.handler import start_server
from tgi_client.utils import read_yaml
from tgi_client.schemas import TGIServerConfig
import os
import signal


#FIXME: use native qwen model loading method
if __name__ == "__main__":
    config_yaml = "/root/tgi_client/config_yamls/qwen.yaml"
    tgi_server_config = TGIServerConfig.parse_obj(read_yaml(config_yaml)["tgi_server_config"])
    proc = start_server(tgi_server_config, "/root/autodl-tmp/logs/qwen.txt")
    try:
        proc.wait()
    except KeyboardInterrupt:
        try:
            proc.terminate()
            print("proc terminated")
        except OSError:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait()
    