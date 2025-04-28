from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from mxbai_rerank import MxbaiRerankV2

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import ComponentDevice, Secret, deserialize_secrets_inplace
from haystack.utils.hf import deserialize_hf_model_kwargs, resolve_hf_device_map, serialize_hf_model_kwargs


@component
class LocalMixedbreadAIRerankerV2:
    """
    Ranks documents based on their semantic similarity to the query.

    It uses a pre-trained model from the MixedBread AI library to evaluate the relevance of documents to a given query.
    """

    def __init__(
        self,
        model: Union[str, Path] = "mixedbread-ai/mxbai-rerank-base-v2",
        *,
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        top_k: int = 10,
        max_length: int = 8192,
        batch_size: int = 16,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        score_threshold: Optional[float] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates an instance of MixedBreadAIRanker.

        :param model:
            The ranking model. Pass a local path or the Hugging Face model name of a MxbaiRerankV2 model.
        :param device:
            The device on which the model is loaded. If `None`, overrides the default device.
        :param token:
            The API token to download private models from Hugging Face.
        :param top_k:
            The maximum number of documents to return per query.
        :param max_length:
            The maximum sequence length for the model. This should be a multiple of 8.
        :param meta_fields_to_embed:
            List of metadata fields to embed with the document.
        :param embedding_separator:
            Separator to concatenate metadata fields to the document.
        :param score_threshold:
            Use it to return documents with a score above this threshold only.
        :param model_kwargs:
            Additional keyword arguments for `AutoModelForCausalLM.from_pretrained`
            when loading the model. Refer to specific model documentation for available kwargs.
        :param tokenizer_kwargs:
            Additional keyword arguments for `AutoTokenizer.from_pretrained` when loading the tokenizer.
            Refer to specific model documentation for available kwargs.
        :param batch_size:
            The batch size to use for inference. The higher the batch size, the more memory is required.
            If you run into memory issues, reduce the batch size.
        """
        self.model = model
        self.device = device
        self.top_k = top_k
        self.token = token
        self.max_length = max_length
        self.meta_fields_to_embed = meta_fields_to_embed
        self.embedding_separator = embedding_separator
        self.score_threshold = score_threshold
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs
        self.batch_size = batch_size
        self._torch_model = None

    def warm_up(self):
        """
        Initializes the component.
        """
        resolved_model_kwargs = resolve_hf_device_map(device=self.device, model_kwargs=self.model_kwargs)
        resolved_kwargs = {
            "device": self.device.to_torch_str(),
            "torch_dtype": resolved_model_kwargs.get("torch_dtype", None),
            "max_length": self.max_length,
            "tokenizer_kwargs": self.tokenizer_kwargs or {},
            **resolved_model_kwargs,
        }
        self._torch_model = MxbaiRerankV2(self.model, **resolved_kwargs)

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Ranks the documents based on relevance to the query using MixedBread Reranker.

        :param query:
            The input query to compare the documents to.
        :param documents:
            A list of documents to be ranked.
        :param top_k:
            The maximum number of documents to return.
        :param score_threshold:
            Use it to return documents only with a score above this threshold.

        :returns:
            A dictionary with the following keys:
            - `documents`: A list of documents closest to the query, sorted from most similar to least similar.

        :raises RuntimeError:
            If the model is not loaded because `warm_up()` was not called before.
        """
        if self._torch_model is None:
            raise RuntimeError(
                "The component MixedBreadAIRankerV2 wasn't warmed up. Run 'warm_up()' before calling 'run()'."
            )

        if not documents:
            return {"documents": []}

        top_k = top_k or self.top_k
        score_threshold = score_threshold or self.score_threshold

        texts = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key]
            ]
            text_to_embed = self.embedding_separator.join(meta_values_to_embed + [doc.content or ""])
            texts.append(text_to_embed)

        # Rank and get scores
        ranked_results = self.torch_model.rank(
            query=query, documents=texts, top_k=top_k, return_documents=True, sort=True, batch_size=self.batch_size
        )

        # Create new document list with scores
        scored_docs = []
        for result in ranked_results:
            ranked_index = result.index
            score = result.score
            doc = documents[ranked_index]
            doc.score = score
            scored_docs.append(doc)

        ranked_docs = sorted(scored_docs, key=lambda x: x.score, reverse=True)

        if score_threshold is not None:
            ranked_docs = [doc for doc in ranked_docs if doc.score >= score_threshold]

        return {"documents": ranked_docs}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        serialization_dict = default_to_dict(
            self,
            device=self.device.to_dict() if self.device else None,
            model=self.model,
            token=self.token.to_dict() if self.token else None,
            top_k=self.top_k,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            score_threshold=self.score_threshold,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
            batch_size=self.batch_size,
        )
        serialize_hf_model_kwargs(serialization_dict["init_parameters"]["model_kwargs"])
        return serialization_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MixedBreadAIRankerV2":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        init_params = data["init_parameters"]

        if init_params.get("device") is not None:
            init_params["device"] = ComponentDevice.from_dict(init_params["device"])

        if init_params.get("model_kwargs") is not None:
            deserialize_hf_model_kwargs(init_params["model_kwargs"])

        deserialize_secrets_inplace(init_params, keys=["token"])
        return default_from_dict(cls, data)
