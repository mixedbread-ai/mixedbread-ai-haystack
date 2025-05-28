from typing import Any, Dict, List, Optional
from haystack import Document, component, default_to_dict, default_from_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from mixedbread_ai_haystack.common.client import MixedbreadClient


@component
class MixedbreadReranker(MixedbreadClient):
    """
    Rerank documents using the Mixedbread Reranking API.
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("MXBAI_API_KEY"),
        model: str = "mixedbread-ai/mxbai-rerank-large-v1",
        top_k: int = 10,
        rank_fields: Optional[List[str]] = None,
        return_input: Optional[bool] = False,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        max_retries: Optional[int] = 2,
    ):
        super(MixedbreadReranker, self).__init__(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
        )
        self.model = model
        self.top_k = top_k
        self.rank_fields = rank_fields or []
        self.return_input = return_input

    def to_dict(self) -> Dict[str, Any]:
        client_params = MixedbreadClient.to_dict(self)["init_parameters"]
        return default_to_dict(
            self,
            **client_params,
            model=self.model,
            top_k=self.top_k,
            rank_fields=self.rank_fields,
            return_input=self.return_input,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MixedbreadReranker":
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _prepare_documents_for_reranking(self, documents: List[Document]) -> List[str]:
        """Prepare document texts for reranking."""
        texts = []
        for doc in documents:
            content_parts = []

            if doc.content:
                content_parts.append(doc.content)

            for field in self.rank_fields:
                if doc.meta.get(field):
                    content_parts.append(str(doc.meta[field]))

            texts.append(" ".join(content_parts))

        return texts

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    def run(self, documents: List[Document], query: str) -> Dict[str, Any]:
        if not isinstance(documents, list) or (
            documents and not isinstance(documents[0], Document)
        ):
            raise TypeError("Input must be a list of Haystack Documents.")
        if not documents:
            return {
                "documents": [],
                "meta": {
                    "model": self.model,
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                    "top_k": 0,
                    "object": "list",
                },
            }

        texts_to_rerank = self._prepare_documents_for_reranking(documents)

        response = self.client.rerank(
            model=self.model,
            query=query,
            input=texts_to_rerank,
            top_k=self.top_k,
            return_input=self.return_input,
        )

        reranked_documents = []
        for result in response.data:
            original_doc = documents[result.index]
            original_doc.meta["rerank_score"] = result.score
            reranked_documents.append(original_doc)

        meta = {
            "model": response.model,
            "usage": response.usage.model_dump(),
            "top_k": len(reranked_documents),
            "object": response.object,
        }

        return {"documents": reranked_documents, "meta": meta}

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    async def run_async(self, documents: List[Document], query: str) -> Dict[str, Any]:
        if not isinstance(documents, list) or (
            documents and not isinstance(documents[0], Document)
        ):
            raise TypeError("Input must be a list of Haystack Documents.")
        if not documents:
            return {
                "documents": [],
                "meta": {
                    "model": self.model,
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                    "top_k": 0,
                    "object": "list",
                },
            }

        texts_to_rerank = self._prepare_documents_for_reranking(documents)

        response = await self.async_client.rerank(
            model=self.model,
            query=query,
            input=texts_to_rerank,
            top_k=self.top_k,
            return_input=self.return_input,
        )

        reranked_documents = []
        for result in response.data:
            original_doc = documents[result.index]
            original_doc.meta["rerank_score"] = result.score
            reranked_documents.append(original_doc)

        meta = {
            "model": response.model,
            "usage": response.usage.model_dump(),
            "top_k": len(reranked_documents),
            "object": response.object,
        }

        return {"documents": reranked_documents, "meta": meta}
