from typing import List, Literal, Optional, Union

from pydantic import BaseModel, validator


class Message(BaseModel):
    role: Literal["system", "assistant", "user"]
    content: str

    def __str__(self):
        return self.content


class PromptFormat(BaseModel):
    system: str
    trailing_assistant: str
    user: str

    default_system_message: str = ""

    @validator("system")
    @classmethod
    def check_system(cls, value):
        assert value and (
            "{instruction}" in value
        ), "system must be a string containing '{instruction}'"
        return value

    @validator("user")
    @classmethod
    def check_user(cls, value):
        assert value and (
            "{instruction}" in value
        ), "user must be a string containing '{instruction}'"
        return value

    def generate_prompt(self, input_prompt: str) -> str:
        messages = [
            Message(role="system", content=self.default_system_message),
            Message(role="user", content=input_prompt),
        ]

        prompt = []
        for message in messages:
            if message.role == "system":
                prompt.append(self.system.format(instruction=message.content))
            elif message.role == "user":
                prompt.append(self.user.format(instruction=message.content))
        prompt.append(self.trailing_assistant)
        return "".join(prompt)


class ServerConfig(BaseModel):
    model_name: str
    port: int = 3000
    master_addr: str = "localhost"


class TGIServerConfig(ServerConfig):
    num_shards: int = 1
    max_batch_total_tokens: int = 8192
    max_batch_prefill_tokens: int = 4096
    max_input_length: int = 1024
    max_total_tokens: int = 1512
    quantize: bool = None
    shard_uds_path: str = "/tmp/text-generation-server"


class VllmServerConfig(ServerConfig):
    trust_remote_code: bool = True
    dtype: str = "auto"
    max_model_len: int  # input + output
    tensor_parallel_size: int = 1
    block_size: int = 16
    swap_space: int = 4
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 256
    max_paddings: int = 256
    quantization: Optional[str] = None


class VllmSamplingParams(BaseModel):
    n: int = (1,)
    temperature: float = (1.0,)
    top_p: float = (1.0,)
    top_k: int = (-1,)
    min_p: float = (0.0,)
    use_beam_search: bool = (False,)
    length_penalty: float = (1.0,)
    stop: Optional[Union[str, List[str]]] = (None,)
    stop_token_ids: Optional[List[int]] = (None,)
    max_tokens: int = (16,)


class PipelineConfig(BaseModel):
    prompt_format: PromptFormat
    server_config: ServerConfig
    gen_parameters: dict
