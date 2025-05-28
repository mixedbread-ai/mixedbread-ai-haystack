from typing import Any, Dict, Optional, Union

from haystack import default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from mixedbread import Mixedbread as MixedbreadSDKClient
from mixedbread import AsyncMixedbread as AsyncMixedbreadSDKClient

USER_AGENT_VERSION = "2.1.1"
USER_AGENT = f"mixedbread-ai-haystack/{USER_AGENT_VERSION}"


class MixedbreadClient:
    """
    Shared client for Mixedbread AI API services.
    Handles API key management and SDK client initialization.
    """

    def __init__(
        self,
        api_key: Union[Secret, str] = Secret.from_env_var("MXBAI_API_KEY"),
        base_url: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        max_retries: Optional[int] = 2,
    ):
        if isinstance(api_key, str):
            api_key = Secret.from_token(api_key)

        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

        resolved_api_key = self.api_key.resolve_value()
        if not resolved_api_key:
            raise ValueError(
                "Mixedbread API key not found. Please set the MXBAI_API_KEY environment variable or pass it directly."
            )

        self._client = MixedbreadSDKClient(
            api_key=resolved_api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers={"User-Agent": USER_AGENT},
        )

        self._async_client = AsyncMixedbreadSDKClient(
            api_key=resolved_api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers={"User-Agent": USER_AGENT},
        )

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MixedbreadClient":
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    @property
    def client(self) -> MixedbreadSDKClient:
        return self._client

    @property
    def async_client(self) -> AsyncMixedbreadSDKClient:
        return self._async_client
