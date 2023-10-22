import logging
from abc import ABC

from text_generation import AsyncClient

from .schemas import PipelineConfig
from .utils import read_yaml

logger = logging.getLogger(__name__)


class Pipeline(ABC):
    def __init__(self, pipeline_yaml_file: str, verbose: int = 1):
        self.verbose = verbose
        pipeline_config = PipelineConfig.parse_obj(read_yaml(pipeline_yaml_file))
        self.gen_parameters = pipeline_config.gen_parameters.dict()
        self.gen_parameters["stop_sequences"] = self.gen_parameters.pop("stop")
        self.gen_parameters.pop("details")
        if self.verbose > 0:
            logger.info(f"parameters for every request: {self.gen_parameters}")
        self.prompt_format = pipeline_config.prompt_format
        self.server_config = pipeline_config.tgi_server_config
        self.tgi_client = self.start_client()

    def start_client(
        self,
    ):
        tgi_client = AsyncClient(
            f"http://{self.server_config.master_addr}:{self.server_config.port}"
        )
        return tgi_client

    async def model_predict(self, input_prompt) -> str:
        formatted_input = self.prompt_format.generate_prompt(input_prompt)
        if self.verbose > 1:
            logger.info(f"formatted input prompt: {repr(formatted_input)}")

        response = await self.tgi_client.generate(
            formatted_input, **self.gen_parameters
        )
        return response.generated_text
