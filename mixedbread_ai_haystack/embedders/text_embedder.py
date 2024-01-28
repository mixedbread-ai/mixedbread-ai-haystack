from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from mixedbread_ai import MixedbreadAi, models
from haystack import component, default_to_dict


class MixedBreadAiTextEmbedderMeta(TypedDict):
    """
    TypedDict for the meta attribute in MixedbreadAiTextEmbedder response.

    Attributes:
        model: The name of the model used for the embedding.
        usage: Detailed information about the usage of tokens.
        truncated: Whether the input text was truncated or not (if the text was too long for the model).
    """
    model: str
    usage: models.ModelUsage
    truncated: bool
    normalized: bool



@component
class MixedbreadAiTextEmbedder:
    """
    A component for top open-source embeddings using mixedbread.ai.

    Usage example:
    ```python
    from mixedbread_ai_haystack import MixedbreadAiTextEmbedder

    text_embedder = MixedbreadAiTextEmbedder()
    text_to_embed = "Bread is love, bread is life."

    print(text_embedder.run(text_to_embed))

    # {
    #    'embedding': [0.06069..., -0.123456..., ...],
    #    'meta': {
    #           'model': 'UAE-Large-V1',
    #           'usage': {'prompt_tokens': 420, 'total_tokens': 420}
    #           'truncated': False,
    #           'normalized': True
    #    }
    # }
    """

    def __init__(
            self,
            model: str = "UAE-Large-V1",
            prefix: str = "",
            suffix: str = "",
            normalized: bool = True,
            instruction: Optional[str] = None,
            api_key: Optional[str] = None,
            base_url: Optional[str] = "https://api.mixedbread.ai",
            custom_headers: Optional[Dict[str, str]] = None,
            timeout: Optional[float] = None,
            verify_ssl: Optional[bool] = None,
    ):
        """
        Create a MixedBreadTextEmbedder component.

        :param model: The name of the MixedBread model to use. Check the list of available models on `https://mixedbread.ai/docs/models/embeddings/`
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        :param normalized: Whether to normalize the embeddings or not.
        :param api_key: The MixedBread API key. It can be explicitly provided or automatically read from the
            environment variable MIXEDBREAD_API_KEY (recommended).
        :param base_url: The URL of the MixedBread API.
        :param timeout: Timeout for the MixedBread API request.
        :param verify_ssl: Whether to verify the SSL certificate for the MixedBread API request.
        :param custom_headers: Custom headers to add to the requests sent to the mixedbread.ai API.
        :param instruction: Used to specify the instruction for the model. Can only be used with instruction based models like e5-large-v2
        for example. If not specified, the default instruction for the model will be used.
        """

        self.model_name = model
        self.prefix = prefix
        self.suffix = suffix
        self.normalized = normalized
        self.instruction = instruction
        self._client = MixedbreadAi(
            api_key=api_key,
            base_url=base_url,
            verify_ssl=verify_ssl,
            timeout=timeout,
            raise_for_status=True,
            headers={
                "User-Agent": "@mixedbread-ai/integrations-haystack",
                **(custom_headers or {}),
            }
        )

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model_name}

    def to_dict(self) -> Dict[str, Any]:
        """
        This method overrides the default serializer in order to avoid leaking the `api_key` value passed
        to the constructor.
        """
        return default_to_dict(self,
                               model=self.model_name,
                               prefix=self.prefix,
                               suffix=self.suffix,
                               normalized=self.normalized,
                               instruction=self.instruction)

    @component.output_types(embedding=List[float], meta=Dict[str, Any])
    def run(self, text: str):
        """Embed a string."""
        if not isinstance(text, str):
            msg = (
                "MixedbreadAiTextEmbedder expects a string as an input."
                "In case you want to embed a list of Documents, please use the MixedbreadAiDocumentEmbedder."
            )
            raise TypeError(msg)

        text_to_embed = self.prefix + text + self.suffix
        res = self._client.embeddings(
            model=self.model_name,
            input=text_to_embed,
            instruction=self.instruction,
            normalized=self.normalized,
        )
        if res is None:
            raise ValueError("MixedbreadAiTextEmbedder received an empty response.")
        if "message" in res:
            raise ValueError(
                f"MixedbreadAiTextEmbedder recieved an unexpected response. Code: {res['code']} Message: {res['message']}")

        embedding = res.data[0].embedding
        metadata = MixedBreadAiTextEmbedderMeta(
            model=self.model_name,
            usage=res.usage,
            truncated=res.data[0].truncated,
            normalized=self.normalized,
        )

        return {"embedding": embedding, "meta": metadata}
