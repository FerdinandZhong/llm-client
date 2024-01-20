import logging
from abc import ABC

from text_generation import AsyncClient as TGIAsyncClient

from .client import VllmAsyncClient
from .schemas import PipelineConfig
from .utils import read_yaml

logger = logging.getLogger(__name__)


class Pipeline(ABC):
    def __init__(
        self, pipeline_yaml_file: str, server_type: str = "vllm", verbose: int = 1
    ):
        self.verbose = verbose
        pipeline_config = PipelineConfig.parse_obj(read_yaml(pipeline_yaml_file))
        self.gen_parameters = pipeline_config.gen_parameters
        if self.verbose > 0:
            logger.info("parameters for every request: %s", self.gen_parameters)
        self.prompt_format = pipeline_config.prompt_format
        self.server_config = pipeline_config.server_config
        if server_type == "vllm":
            self.client = self.start_client(VllmAsyncClient)
        else:
            self.client = self.start_client(TGIAsyncClient)

    def start_client(self, client_cls):
        """
        Starts a client based on the provided client class.

        Args:
            client_cls (Class): The class of the client to start.

        Returns:
            An instance of the client class.
        """
        return client_cls(
            f"http://{self.server_config.master_addr}:{self.server_config.port}"
        )

    async def model_predict(self, input_prompt) -> str:
        """
        Predicts the model output based on the given input prompt.

        Parameters:
        input_prompt (str): The input prompt for the prediction.

        Returns:
        str: The predicted output from the model.
        """
        formatted_input = self.prompt_format.generate_prompt(input_prompt)
        if self.verbose > 1:
            logger.info("formatted input prompt: %r", repr(formatted_input))

        response = await self.client.generate(formatted_input, **self.gen_parameters)
        return response.generated_text
