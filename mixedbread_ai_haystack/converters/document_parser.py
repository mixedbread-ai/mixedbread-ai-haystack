import time
from typing import Optional, List, Dict, Any, Union, Literal
from pathlib import Path

from haystack import component, Document, default_to_dict, default_from_dict
from haystack.dataclasses import ByteStream
from haystack.utils import Secret, deserialize_secrets_inplace

from mixedbread_ai_haystack.common.client import MixedbreadClient


@component
class MixedbreadDocumentParser(MixedbreadClient):
    """
    Parses various file types (PDF, DOCX, images, etc.) using Mixedbread parsing jobs.
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("MXBAI_API_KEY"),
        chunking_strategy: Literal["page", "paragraph", "sentence"] = "page",
        return_format: Literal["markdown", "text"] = "markdown",
        element_types: Optional[List[str]] = None,
        include_page_breaks: bool = True,
        max_wait_time: int = 300,
        poll_interval: int = 5,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        max_retries: Optional[int] = 2,
        store_full_path: bool = False,
    ):
        """
        Create a MixedbreadDocumentParser component.
        """
        super(MixedbreadDocumentParser, self).__init__(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
        )

        self.chunking_strategy = chunking_strategy
        self.return_format = return_format
        self.element_types = element_types or ["text", "title", "list-item", "table"]
        self.include_page_breaks = include_page_breaks
        self.max_wait_time = max_wait_time
        self.poll_interval = poll_interval
        self.store_full_path = store_full_path

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the component to a dictionary."""
        client_params = MixedbreadClient.to_dict(self)["init_parameters"]
        return default_to_dict(
            self,
            **client_params,
            chunking_strategy=self.chunking_strategy,
            return_format=self.return_format,
            element_types=self.element_types,
            include_page_breaks=self.include_page_breaks,
            max_wait_time=self.max_wait_time,
            poll_interval=self.poll_interval,
            store_full_path=self.store_full_path,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MixedbreadDocumentParser":
        """Deserialize the component from a dictionary."""
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _upload_file(self, file_path: Union[str, Path, ByteStream]) -> str:
        """Upload a file to Mixedbread AI and return the file ID."""
        if isinstance(file_path, ByteStream):
            # For ByteStream objects, we need to write to a temporary file
            # or pass the data directly if the API supports it
            content = file_path.data
            filename = file_path.meta.get("file_path", "document")

            # Create a file-like object from ByteStream
            from io import BytesIO

            file_obj = BytesIO(content)
            file_obj.name = filename

            result = self.client.files.create(file=file_obj)
        else:
            # Handle file paths
            with open(file_path, "rb") as f:
                result = self.client.files.create(file=f)

        return result.id

    def _create_parsing_job(self, file_id: str) -> str:
        """Create a parsing job and return the job ID."""
        result = self.client.parsing.jobs.create(
            file_id=file_id,
            chunking_strategy=self.chunking_strategy,
            return_format=self.return_format,
            element_types=self.element_types,
        )
        return result.id

    def _wait_for_job_completion(self, job_id: str) -> Dict[str, Any]:
        """Wait for the parsing job to complete and return the result."""
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
        """Convert parsing results into Haystack Documents."""
        documents = []
        result_data = parsing_result.get("result", {})
        chunks = result_data.get("chunks", [])

        # Get source information for metadata
        if isinstance(source_path, ByteStream):
            source_info = source_path.meta.get("file_path", "ByteStream")
            file_name = (
                Path(source_info).name if source_info != "ByteStream" else "ByteStream"
            )
        else:
            source_info = str(source_path)
            file_name = Path(source_path).name

        base_metadata = {
            "file_path": source_info if self.store_full_path else file_name,
            "parsing_job_id": parsing_result.get("id"),
            "chunking_strategy": result_data.get("chunking_strategy"),
            "return_format": result_data.get("return_format"),
            "element_types": result_data.get("element_types"),
            "page_sizes": result_data.get("page_sizes"),
            "total_chunks": len(chunks),
        }

        # Add user-provided metadata
        if meta:
            base_metadata.update(meta)

        for i, chunk in enumerate(chunks):
            chunk_content = chunk.get("content", "")
            content_to_embed = chunk.get("content_to_embed", chunk_content)
            elements = chunk.get("elements", [])

            chunk_metadata = base_metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_index": i,
                    "elements": elements,
                    "element_count": len(elements),
                    "content_to_embed": content_to_embed,
                }
            )

            # Add element-specific metadata
            if elements:
                element_types_in_chunk = list(
                    set(elem.get("type") for elem in elements)
                )
                chunk_metadata["element_types_in_chunk"] = element_types_in_chunk

                # Add page information if available
                pages = list(
                    set(
                        elem.get("page")
                        for elem in elements
                        if elem.get("page") is not None
                    )
                )
                if pages:
                    chunk_metadata["pages"] = sorted(pages)
                    chunk_metadata["page_range"] = (
                        f"{min(pages)}-{max(pages)}"
                        if len(pages) > 1
                        else str(pages[0])
                    )

            document = Document(content=chunk_content, meta=chunk_metadata)
            documents.append(document)

        return documents

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        """
        Parse files using Mixedbread AI's parsing service.

        Args:
            sources: List of file paths or ByteStream objects to parse
            meta: Optional metadata to attach to the documents

        Returns:
            Dictionary with "documents" key containing the parsed documents
        """
        if not sources:
            return {"documents": []}

        # Handle metadata
        if meta is None:
            meta_list = [{}] * len(sources)
        elif isinstance(meta, dict):
            meta_list = [meta] * len(sources)
        elif isinstance(meta, list):
            if len(meta) != len(sources):
                raise ValueError(
                    f"Length of meta list ({len(meta)}) must match length of sources ({len(sources)})"
                )
            meta_list = meta
        else:
            raise TypeError("meta must be a dict, list of dicts, or None")

        all_documents = []

        for source, source_meta in zip(sources, meta_list):
            try:
                # Upload file
                file_id = self._upload_file(source)

                # Create parsing job
                job_id = self._create_parsing_job(file_id)

                # Wait for completion and get results
                parsing_result = self._wait_for_job_completion(job_id)

                # Convert to documents
                documents = self._create_documents_from_result(
                    parsing_result, source, source_meta
                )
                all_documents.extend(documents)

            except Exception as e:
                # Add error handling - you might want to skip failed files or raise
                error_msg = f"Failed to parse {source}: {str(e)}"
                print(f"Warning: {error_msg}")

                # Create an error document so the user knows what failed
                error_doc = Document(
                    content="",
                    meta={
                        "file_path": (
                            str(source)
                            if not isinstance(source, ByteStream)
                            else "ByteStream"
                        ),
                        "parsing_error": error_msg,
                        "parsing_status": "failed",
                        **(source_meta or {}),
                    },
                )
                all_documents.append(error_doc)

        return {"documents": all_documents}

    @component.output_types(documents=List[Document])
    async def run_async(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        """
        Async version of the run method.

        Note: This implementation uses sync calls but could be optimized
        for true async operation with concurrent job processing.
        """
        # For now, we'll use the sync implementation
        # In a future version, this could be optimized to run multiple
        # parsing jobs concurrently
        return self.run(sources=sources, meta=meta)
