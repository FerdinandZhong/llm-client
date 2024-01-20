from typing import Dict, Optional

from aiohttp import ClientSession, ClientTimeout
from text_generation.errors import parse_error
from text_generation.types import Response


class VllmAsyncClient:
    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        timeout: int = 10,
    ):
        """
        Args:
            base_url (`str`):
                text-generation-inference instance base url
            headers (`Optional[Dict[str, str]]`):
                Additional headers
            cookies (`Optional[Dict[str, str]]`):
                Cookies to include in the requests
            timeout (`int`):
                Timeout in seconds
        """
        self.base_url = base_url
        self.headers = headers
        self.cookies = cookies
        self.timeout = ClientTimeout(timeout * 60)

    async def generate(self, prompt: str, **sampling_params):
        """
        Generate text based on the given prompt and sampling parameters.

        Args:
            prompt (str): The prompt to generate text from.
            sampling_params (dict): Additional parameters for sampling.

        Returns:
            Response: A response object containing the generated text.
        """
        async with ClientSession(
            headers=self.headers, cookies=self.cookies, timeout=self.timeout
        ) as session:
            request_dict = {"prompt": prompt, **sampling_params}
            async with session.post(
                f"{self.base_url}/generate", json=request_dict
            ) as resp:
                payload = await resp.json()

                if resp.status != 200:
                    raise parse_error(resp.status, payload)
                return Response(generated_text=payload["text"], details=None)
