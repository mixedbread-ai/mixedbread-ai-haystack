from typing import Any, Dict, List, Optional, TypedDict

from haystack import Document, component, default_to_dict, logging
from mixedbread_ai import Usage, ObjectType

from mixedbread_ai_haystack.common.client import MixedbreadAIClient

logger = logging.getLogger(__name__)


class RerankerMeta(TypedDict, total=False):
    usage: Usage
    model: str
    object: ObjectType
    top_k: int


@component
class MixedbreadAIReranker(MixedbreadAIClient):
    """
    Ranks Documents based on their similarity to the query using mixedbread ai's reranking API.

    Documents are indexed from most to least semantically relevant to the query.

    Find out more at https://mixedbread.ai/docs

    Usage example:
    ```python
    from haystack import Document
    from mixedbread_ai_haystack import MixedbreadAIReranker

    ranker = MixedbreadAIReranker(model="mixedbread-ai/mxbai-rerank-large-v1", top_k=2)

    docs = [Document(content="Paris"), Document(content="Berlin")]
    query = "What is the capital of Germany?"
    output = ranker.run(query=query, documents=docs)
    docs = output["documents"]
    ```
    """

    def __init__(
        self,
        model: str = "default",
        top_k: int = 20,
        meta_fields_to_rank: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initializes an instance of the 'MixedbreadAIReranker'.

        Parameters:
            model (str): Mixedbread AI model name.
            top_k (int): The maximum number of documents to return.
            meta_fields_to_rank (Optional[List[str]]): List of meta fields that should be concatenated
                with the document content for reranking.
        """
        super(MixedbreadAIReranker, self).__init__(**kwargs)

        self.model = model
        self.top_k = top_k
        self.meta_fields_to_rank = meta_fields_to_rank or []

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary with serialized data.
        """
        parent_params = super(MixedbreadAIReranker, self).to_dict()["init_parameters"]

        return default_to_dict(
            self,
            **parent_params,
            model=self.model,
            top_k=self.top_k,
            meta_fields_to_rank=self.meta_fields_to_rank,
        )

    @component.output_types(documents=List[Document], meta=RerankerMeta)
    def run(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Uses the Mixedbread Reranker to re-rank the list of documents based on the query.

        Parameters:
            query (str): Query string.
            documents (List[Document]): List of Documents to be ranked.
            top_k (Optional[int]): The maximum number of Documents you want the Ranker to return.

        Returns:
            Dict[str, Any]: A dictionary containing the ranked documents with the following key:
                - `documents`: List of Documents most similar to the given query in descending order of similarity.

        Raises:
            ValueError: If `top_k` is not > 0.
        """
        top_k = top_k or self.top_k
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        dicts = [doc.to_dict() for doc in documents]
        rank_fields = list({*self.meta_fields_to_rank, "content"})

        response = self._client.reranking(
            model=self.model,
            query=query,
            input=dicts,
            rank_fields=rank_fields,
            top_k=top_k,
            return_input=False,
            request_options=self._request_options
        )

        sorted_docs = []
        for result in response.data:
            doc = documents[result.index]
            doc.score = result.score
            sorted_docs.append(doc)

        return {
            "documents": sorted_docs,
            "meta": RerankerMeta(
                **response.dict(exclude={"data", "usage", "return_input"}),
                usage=response.usage
            )
        }
