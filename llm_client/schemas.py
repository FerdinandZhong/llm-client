from typing import Literal

from pydantic import BaseModel, validator
from text_generation.types import Parameters


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
    def check_system(cls, value):
        assert value and (
            "{instruction}" in value
        ), "system must be a string containing '{instruction}'"
        return value

    @validator("user")
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


class TGIServerConfig(BaseModel):
    model_name: str
    num_shards: int = 1
    port: int = 3000
    master_addr: str = "localhost"
    max_batch_total_tokens: int = 8192
    max_batch_prefill_tokens: int = 4096
    max_input_length: int = 1024
    max_total_tokens: int = 1512
    quantize: bool = None
    shard_uds_path: str = "/tmp/text-generation-server"


class PipelineConfig(BaseModel):
    prompt_format: PromptFormat
    tgi_server_config: TGIServerConfig
    gen_parameters: Parameters = Parameters()
