import logging
import subprocess

from .schemas import TGIServerConfig, VllmServerConfig

logger = logging.getLogger(__name__)


def start_tgi_server(
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
        tgi_server_command += f"2>&1 | tee -a {tgi_server_log_file}"
        tgi_bash_command = (
            f"! /bin/bash -c 'source activate {tgi_server_env}; {tgi_server_command}'"
        )

        logger.info(f"full tgi server command: {tgi_bash_command}")
        proc = subprocess.Popen(tgi_bash_command, shell=True)
        return proc
    except Exception as ex:
        logger.error(str(ex))


def start_vllm_server(
    server_config: VllmServerConfig,
    vllm_server_log_file: str = "/root/autodl-tmp/logs",
):
    try:
        logger.info("server full config: %s", server_config.dict())
        vllm_server_command = (
            "python -m vllm.entrypoints.api_server "
            f"--model {server_config.model_name} --tensor-parallel-size {server_config.tensor_parallel_size} "
            f"--dtype {server_config.dtype} --max-model-len {server_config.max_model_len} "
            f"--block-size {server_config.block_size} --gpu-memory-utilization {server_config.gpu_memory_utilization} "
            f"--max-num-seqs {server_config.max_num_seqs} --max-paddings {server_config.max_paddings} "   
        )
        vllm_server_command += (
            f"--quantization {server_config.quantization} " if server_config.quantization else ""
        )

        vllm_server_command += (
            "--trust-remote-code" if server_config.trust_remote_code else ""
        )
        vllm_server_command += f"2>&1 | tee -a {vllm_server_log_file}"
        proc = subprocess.Popen(vllm_server_command, shell=True)
        return proc
    except Exception as ex:
        logger.error(str(ex))

