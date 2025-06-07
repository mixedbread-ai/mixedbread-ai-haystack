import time
import asyncio
from typing import Optional, List, Dict, Any, Union, Literal
from pathlib import Path

from haystack import component, Document, default_to_dict, default_from_dict
from haystack.dataclasses import ByteStream
from haystack.utils import Secret, deserialize_secrets_inplace

from mixedbread_ai_haystack.common.client import MixedbreadClient
from mixedbread_ai_haystack.common.logging import get_logger

logger = get_logger(__name__)


@component
class MixedbreadDocumentParser(MixedbreadClient):
    """
    Parse documents using the Mixedbread Parsing API.
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("MXBAI_API_KEY"),
        chunking_strategy: Optional[str] = "page",
        return_format: Literal["markdown", "plain"] = "markdown",
        element_types: Optional[List[str]] = None,
        max_wait_time: int = 300,
        poll_interval: int = 5,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        max_retries: Optional[int] = 2,
    ):
        """
        Initialize the MixedbreadDocumentParser.
        
        Args:
            api_key: Mixedbread API key.
            chunking_strategy: Strategy for chunking documents ("page", "semantic", etc.).
            return_format: Output format ("markdown" or "plain").
            element_types: Types of elements to extract.
            max_wait_time: Maximum time to wait for parsing completion.
            poll_interval: Interval between polling for job status.
            base_url: Optional custom API base URL.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
        """
        super(MixedbreadDocumentParser, self).__init__(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
        )

        self.chunking_strategy = chunking_strategy
        self.return_format = return_format
        self.element_types = element_types or ["text", "title", "list-item", "table"]
        self.max_wait_time = max_wait_time
        self.poll_interval = poll_interval

    def to_dict(self) -> Dict[str, Any]:
        client_params = MixedbreadClient.to_dict(self)["init_parameters"]
        return default_to_dict(
            self,
            **client_params,
            chunking_strategy=self.chunking_strategy,
            return_format=self.return_format,
            element_types=self.element_types,
            max_wait_time=self.max_wait_time,
            poll_interval=self.poll_interval,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MixedbreadDocumentParser":
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _get_file_name(self, source: Union[str, Path, ByteStream]) -> str:
        """
        Extract file name from various source types.
        
        Args:
            source: File path string, Path object, or ByteStream.
            
        Returns:
            Extracted filename.
        """
        if isinstance(source, ByteStream):
            file_path = source.meta.get("file_path", "ByteStream")
            return Path(file_path).name if file_path != "ByteStream" else "ByteStream"
        return Path(source).name

    def _upload_file(self, file_path: Union[str, Path, ByteStream]) -> str:
        if isinstance(file_path, ByteStream):
            content = file_path.data
            filename = file_path.meta.get("file_path", "document")

            from io import BytesIO

            file_obj = BytesIO(content)
            file_obj.name = filename

            result = self.client.files.create(file=file_obj)
        else:
            with open(file_path, "rb") as f:
                result = self.client.files.create(file=f)

        return result.id

    def _create_parsing_job(self, file_id: str) -> str:
        result = self.client.parsing.jobs.create(
            file_id=file_id,
            chunking_strategy=self.chunking_strategy,
            return_format=self.return_format,
            element_types=self.element_types,
        )
        return result.id

    def _wait_for_job_completion(self, job_id: str) -> Dict[str, Any]:
        start_time = time.time()

        while time.time() - start_time < self.max_wait_time:
            result = self.client.parsing.jobs.retrieve(job_id=job_id)

            if result.status == "completed":
                return result.model_dump()
            elif result.status == "failed":
                error_msg = result.error or "Unknown parsing error"
                raise RuntimeError(f"Parsing job failed: {error_msg}")

            time.sleep(self.poll_interval)

        raise TimeoutError(
            f"Parsing job {job_id} did not complete within {self.max_wait_time} seconds"
        )

    def _create_documents_from_result(
        self,
        parsing_result: Dict[str, Any],
        source_path: Union[str, Path, ByteStream],
        meta: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        result_data = parsing_result.get("result", {})
        chunks = result_data.get("chunks", [])

        base_metadata = {
            "file_path": self._get_file_name(source_path),
            "parsing_job_id": parsing_result.get("id"),
            "chunking_strategy": result_data.get("chunking_strategy"),
            "return_format": result_data.get("return_format"),
            "element_types": result_data.get("element_types"),
            "page_sizes": result_data.get("page_sizes"),
            "total_chunks": len(chunks),
            **(meta or {}),
        }

        documents = []
        for i, chunk in enumerate(chunks):
            chunk_content = chunk.get("content", "")
            elements = chunk.get("elements", [])

            chunk_metadata = {
                **base_metadata,
                "chunk_index": i,
                "elements": elements,
                "element_count": len(elements),
                "content_to_embed": chunk.get("content_to_embed", chunk_content),
            }

            if elements:
                chunk_metadata["element_types_in_chunk"] = list(
                    set(elem.get("type") for elem in elements)
                )

                pages = [
                    elem.get("page")
                    for elem in elements
                    if elem.get("page") is not None
                ]
                if pages:
                    unique_pages = sorted(set(pages))
                    chunk_metadata["pages"] = unique_pages
                    chunk_metadata["page_range"] = (
                        f"{min(unique_pages)}-{max(unique_pages)}"
                        if len(unique_pages) > 1
                        else str(unique_pages[0])
                    )

            documents.append(Document(content=chunk_content, meta=chunk_metadata))

        return documents

    def _create_error_document(
        self, source: Union[str, Path, ByteStream], error_msg: str, meta: Dict[str, Any]
    ) -> Document:
        """Create an error document when parsing fails."""
        return Document(
            content="",
            meta={
                "file_path": self._get_file_name(source),
                "parsing_error": error_msg,
                "parsing_status": "failed",
                **meta,
            },
        )

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        """
        Parse documents from various sources.
        
        Args:
            sources: List of file paths or ByteStream objects to parse.
            meta: Optional metadata to attach to documents.
            
        Returns:
            Dictionary containing parsed documents.
        """
        if not sources:
            return {"documents": []}

        # Normalize meta to a list
        if meta is None:
            meta_list = [{}] * len(sources)
        elif isinstance(meta, dict):
            meta_list = [meta] * len(sources)
        else:
            if len(meta) != len(sources):
                raise ValueError(
                    f"Length of meta list ({len(meta)}) must match length of sources ({len(sources)})"
                )
            meta_list = meta

        all_documents = []
        for source, source_meta in zip(sources, meta_list):
            try:
                file_id = self._upload_file(source)
                job_id = self._create_parsing_job(file_id)
                parsing_result = self._wait_for_job_completion(job_id)
                documents = self._create_documents_from_result(
                    parsing_result, source, source_meta
                )
                all_documents.extend(documents)

            except Exception as e:
                error_msg = f"Failed to parse {source}: {str(e)}"
                logger.warning(error_msg)
                error_doc = self._create_error_document(source, error_msg, source_meta)
                all_documents.append(error_doc)

        return {"documents": all_documents}

    async def _process_file_async(
        self, source: Union[str, Path, ByteStream], meta: Dict[str, Any]
    ) -> List[Document]:
        """Process a single file asynchronously."""
        try:
            # Run the sync operations in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Upload file
            file_id = await loop.run_in_executor(None, self._upload_file, source)
            
            # Create parsing job
            job_id = await loop.run_in_executor(None, self._create_parsing_job, file_id)
            
            # Wait for completion
            parsing_result = await loop.run_in_executor(
                None, self._wait_for_job_completion, job_id
            )
            
            # Create documents
            documents = await loop.run_in_executor(
                None, self._create_documents_from_result, parsing_result, source, meta
            )
            
            return documents
            
        except Exception as e:
            error_msg = f"Failed to parse {source}: {str(e)}"
            logger.warning(error_msg)
            error_doc = self._create_error_document(source, error_msg, meta)
            return [error_doc]

    @component.output_types(documents=List[Document])
    async def run_async(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        """
        Asynchronously parse documents from various sources using concurrent processing.
        
        Args:
            sources: List of file paths or ByteStream objects to parse.
            meta: Optional metadata to attach to documents.
            
        Returns:
            Dictionary containing parsed documents.
        """
        """Process multiple files concurrently using asyncio.gather."""
        if not sources:
            return {"documents": []}

        # Normalize meta to a list
        if meta is None:
            meta_list = [{}] * len(sources)
        elif isinstance(meta, dict):
            meta_list = [meta] * len(sources)
        else:
            if len(meta) != len(sources):
                raise ValueError(
                    f"Length of meta list ({len(meta)}) must match length of sources ({len(sources)})"
                )
            meta_list = meta

        # Process files concurrently
        tasks = [
            self._process_file_async(source, source_meta)
            for source, source_meta in zip(sources, meta_list)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and handle exceptions
        all_documents = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = f"Failed to process {sources[i]}: {str(result)}"
                logger.warning(error_msg)
                error_doc = self._create_error_document(sources[i], error_msg, meta_list[i])
                all_documents.append(error_doc)
            else:
                all_documents.extend(result)

        return {"documents": all_documents}
