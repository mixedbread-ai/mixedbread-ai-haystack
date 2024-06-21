from typing import Any, Dict, List, Optional, TypedDict

from haystack import component, default_to_dict
from mixedbread_ai import EncodingFormat, TruncationStrategy, Usage, ObjectType

from mixedbread_ai_haystack.common.client import MixedbreadAIClient


class EmbedderMeta(TypedDict):
    usage: Usage
    model: str
    object: ObjectType
    normalized: bool
    encoding_format: EncodingFormat
    dimensions: int


@component
class MixedbreadAITextEmbedder(MixedbreadAIClient):
    """
    A component for generating text embeddings using mixedbread ai's embedding API.

    Find out more at https://mixedbread.ai/docs

    To use this you'll need a mixedbread ai API key - either pass it to
    the api_key parameter or set the MXBAI_API_KEY environment variable.

    API keys are available on https://mixedbread.ai - it's free to sign up and trial API
    keys work with this implementation.

    Usage example:
        ```python
        from mixedbread_ai_haystack import MixedbreadAITextEmbedder

        text_embedder = MixedbreadAITextEmbedder(
            model="mixedbread-ai/mxbai-embed-large-v1"
        )

        text_to_embed = "Bread is love, bread is life."

        print(text_embedder.run(text_to_embed))
        ```

    Attributes:
        model (str): The model to use for generating embeddings.
        prefix (str): The prefix to add to the text before embedding.
        suffix (str): The suffix to add to the text before embedding.
        normalized (bool): Whether to normalize the embeddings.
        encoding_format (EncodingFormat): The format for encoding the embeddings.
        truncation_strategy (TruncationStrategy): The strategy for truncating the text.
        dimensions (Optional[int]): The desired number of dimensions in the output vectors.
            Only applicable for Matryoshka-based models.
        prompt (Optional[str]): The prompt to use for the embedding model.
    """

    def __init__(
            self,
            model: str = "mixedbread-ai/mxbai-embed-large-v1",
            prefix: str = "",
            suffix: str = "",
            normalized: bool = True,
            encoding_format: EncodingFormat = EncodingFormat.FLOAT,
            truncation_strategy: TruncationStrategy = TruncationStrategy.START,
            dimensions: Optional[int] = None,
            prompt: Optional[str] = None,
            **kwargs
    ):
        super(MixedbreadAITextEmbedder, self).__init__(**kwargs)

        self.model = model
        self.prefix = prefix
        self.suffix = suffix
        self.normalized = normalized
        self.encoding_format = encoding_format
        self.truncation_strategy = truncation_strategy
        self.dimensions = dimensions
        self.prompt = prompt

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        Returns:
            Dict[str, Any]: The serialized component data.
        """
        parent_params = super(MixedbreadAITextEmbedder, self).to_dict()["init_parameters"]

        return default_to_dict(self,
                               **parent_params,
                               model=self.model,
                               prefix=self.prefix,
                               suffix=self.suffix,
                               normalized=self.normalized,
                               encoding_format=self.encoding_format,
                               truncation_strategy=self.truncation_strategy,
                               dimensions=self.dimensions,
                               prompt=self.prompt
                               )

    @component.output_types(embedding=List[float], meta=EmbedderMeta)
    def run(self, text: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Embeds a string of text and returns the embedding and metadata.

        Parameters:
            text (str): The text to embed.
            prompt (Optional[str]): An optional prompt to use with the embedding model.

        Returns:
            Dict[str, Any]: A dictionary with the following keys:
                - `embedding`: The embedding of the input text.
                - `meta`: Metadata about the request.

        Raises:
            TypeError: If the input is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(
                "MixedbreadAITextEmbedder expects a string as an input. "
                "In case you want to embed a list of Documents, please use the MixedbreadAIDocumentEmbedder."
            )

        text_to_embed = self.prefix + text + self.suffix
        response = self._client.embeddings(
            model=self.model,
            input=text_to_embed,
            normalized=self.normalized,
            encoding_format=self.encoding_format,
            truncation_strategy=self.truncation_strategy,
            dimensions=self.dimensions,
            prompt=prompt or self.prompt,
            request_options=self._request_options
        )

        return {
            "embedding": response.data[0].embedding,
            "meta": EmbedderMeta(
                **response.dict(exclude={"data", "usage"}),
                usage=response.usage
            )
        }
