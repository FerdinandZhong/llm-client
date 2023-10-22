import logging
import subprocess

from .schemas import TGIServerConfig

logger = logging.getLogger(__name__)


def start_server(
    server_config: TGIServerConfig,
    tgi_server_log_file: str = "/root/autodl-tmp/logs",
    tgi_server_env: str = "llm-tgi-env",
):
    try:
        logger.info(f"server full config: {server_config.dict()}")
        tgi_server_command = (
            f"text-generation-launcher --model-id {server_config.model_name} "
            f"--port {server_config.port} --master-addr {server_config.master_addr} "
            f"--num-shard {server_config.num_shards} --shard-uds-path {server_config.shard_uds_path} "
            f"--max-batch-total-tokens {server_config.max_batch_total_tokens} "
            f"--max-batch-prefill-tokens {server_config.max_batch_prefill_tokens} "
            f"--max-input-length {server_config.max_input_length} --max-total-tokens {server_config.max_total_tokens} "
        )
        tgi_server_command += (
            f"--quantize {server_config.quantize}" if server_config.quantize else ""
        )
        tgi_server_command += (
            f"2>&1 | tee -a {tgi_server_log_file}"
        )
        tgi_bash_command = (
            f"! /bin/bash -c 'source activate {tgi_server_env}; {tgi_server_command}'"
        )

        logger.info(f"full tgi server command: {tgi_bash_command}")
        proc = subprocess.Popen(tgi_bash_command, shell=True)
        return proc
    except Exception as ex:
        logger.error(str(ex))
