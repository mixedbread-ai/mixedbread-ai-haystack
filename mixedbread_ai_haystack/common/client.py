from typing import Any, Dict, Optional, Union

import httpx
from haystack import default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from mixedbread_ai.client import MixedbreadAI, AsyncMixedbreadAI
from mixedbread_ai.core import RequestOptions

USER_AGENT = "mixedbread-ai@haystack/2.0.0"


class MixedbreadAIClient:
    """
    mixedbread ai Client configuration and initialization.

    Attributes:
        api_key (Secret): mixedbread ai API key. Must be specified directly or via environment variable 'MXBAI_API_KEY'.
        base_url (Optional[str]): Base URL for the mixedbread ai API. Leave blank if not using a proxy or service emulator.
        timeout (Optional[float]): Timeout for the mixedbread ai API.
        max_retries (Optional[int]): Max retries for the mixedbread ai API.
        httpx_client (Optional[httpx.Client]): An optional synchronous HTTPX client instance (not serialized).
    """

    def __init__(
            self,
            api_key: Union[Secret, str] = Secret.from_env_var("MXBAI_API_KEY"),
            base_url: Optional[str] = None,
            timeout: Optional[float] = 60.0,
            max_retries: Optional[int] = 3,
            httpx_client: Optional[httpx.Client] = None,  # not serialized
    ):
        if api_key is None:
            raise ValueError(
                "The mixedbread ai API key must be specified."
                + "You either pass it in the constructor using 'api_key'"
                + "or via the 'MXBAI_API_KEY' environment variable."
            )
        if isinstance(api_key, str):
            api_key = Secret.from_token(api_key)
        if timeout is not None and timeout <= 0:
            raise ValueError("The timeout parameter must be greater than 0.")
        if max_retries is not None and max_retries < 0:
            raise ValueError("The max_retries parameter must be greater than or equal to 0.")

        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

        self._request_options = RequestOptions(
            max_retries=max_retries,
            timeout=timeout,
            additional_headers={"User-Agent": USER_AGENT},
        )
        self._client = MixedbreadAI(
            api_key=api_key.resolve_value(),
            base_url=base_url,
            httpx_client=httpx_client
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the client to a dictionary.

        Returns:
            Dict[str, Any]: The serialized client.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Any:
        """
        Deserializes the component from a dictionary.

        Parameters:
            data (Dict[str, Any]): Dictionary to deserialize from.

        Returns:
            Any: The deserialized component.
        """
        init_params = data.get("init_parameters", {})
        deserialize_secrets_inplace(init_params, ["api_key"])
        return default_from_dict(cls, data)
