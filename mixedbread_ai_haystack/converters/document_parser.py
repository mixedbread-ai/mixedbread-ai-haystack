import asyncio
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.dataclasses import ByteStream
from haystack.utils import Secret, deserialize_secrets_inplace
from mixedbread import AsyncMixedbread, Mixedbread


@component
class MixedbreadDocumentParser:
    """
    Parse documents using the Mixedbread Parsing API.

    Supports both synchronous and asynchronous parsing with concurrent processing.
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("MXBAI_API_KEY"),
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
            return_format: Output format ("markdown" or "plain").
            element_types: Types of elements to extract. Available types: caption, footnote, 
                          formula, list-item, page-footer, page-header, picture, section-header, 
                          table, text, title.
            max_wait_time: Maximum time to wait for parsing completion.
            poll_interval: Interval between polling for job status.
            base_url: Optional custom API base URL.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
        """
        resolved_api_key = api_key.resolve_value()
        if not resolved_api_key:
            raise ValueError(
                "Mixedbread API key not found. Set MXBAI_API_KEY environment variable."
            )

        self.api_key = api_key
        self.return_format = return_format
        self.element_types = element_types or ["text", "title", "list-item", "table", "section-header"]
        self.max_wait_time = max_wait_time
        self.poll_interval = poll_interval
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize clients
        self.client = Mixedbread(
            api_key=resolved_api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

        self.aclient = AsyncMixedbread(
            api_key=resolved_api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the parser configuration."""
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            return_format=self.return_format,
            element_types=self.element_types,
            max_wait_time=self.max_wait_time,
            poll_interval=self.poll_interval,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MixedbreadDocumentParser":
        """Create parser from dictionary."""
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _get_filename(self, source: Union[str, Path, ByteStream]) -> str:
        """Extract filename from source."""
        if isinstance(source, ByteStream):
            return source.meta.get("file_path", "ByteStream")
        return str(Path(source).name)

    def _upload_file(self, source: Union[str, Path, ByteStream]) -> str:
        """Upload file and return file ID."""
        if isinstance(source, ByteStream):
            file_obj = BytesIO(source.data)
            file_obj.name = self._get_filename(source)
            return self.client.files.create(file=file_obj).id
        else:
            with open(source, "rb") as f:
                return self.client.files.create(file=f).id

    async def _upload_file_async(self, source: Union[str, Path, ByteStream]) -> str:
        """Upload file asynchronously and return file ID."""
        if isinstance(source, ByteStream):
            file_obj = BytesIO(source.data)
            file_obj.name = self._get_filename(source)
            return (await self.aclient.files.create(file=file_obj)).id
        else:
            with open(source, "rb") as f:
                return (await self.aclient.files.create(file=f)).id

    def _wait_for_completion(self, job_id: str) -> Dict[str, Any]:
        """Wait for parsing job completion."""
        start_time = time.time()

        while time.time() - start_time < self.max_wait_time:
            result = self.client.parsing.jobs.retrieve(job_id=job_id)

            if result.status == "completed":
                return result.model_dump()
            elif result.status == "failed":
                raise RuntimeError(f"Parsing failed: {result.error or 'Unknown error'}")

            time.sleep(self.poll_interval)

        raise TimeoutError(
            f"Parsing job {job_id} timed out after {self.max_wait_time}s"
        )

    async def _wait_for_completion_async(self, job_id: str) -> Dict[str, Any]:
        """Wait for parsing job completion asynchronously."""
        start_time = time.time()

        while time.time() - start_time < self.max_wait_time:
            result = await self.aclient.parsing.jobs.retrieve(job_id=job_id)

            if result.status == "completed":
                return result.model_dump()
            elif result.status == "failed":
                raise RuntimeError(f"Parsing failed: {result.error or 'Unknown error'}")

            await asyncio.sleep(self.poll_interval)

        raise TimeoutError(
            f"Parsing job {job_id} timed out after {self.max_wait_time}s"
        )

    def _create_documents(
        self, parsing_result: Dict[str, Any], source: Union[str, Path, ByteStream]
    ) -> List[Document]:
        """Create Document objects from parsing result."""
        result_data = parsing_result.get("result", {})
        chunks = result_data.get("chunks", [])

        documents = []
        for i, chunk in enumerate(chunks):
            content = chunk.get("content", "")
            elements = chunk.get("elements", [])

            # Extract page information
            pages = [
                elem.get("page") for elem in elements if elem.get("page") is not None
            ]
            unique_pages = sorted(set(pages)) if pages else []

            metadata = {
                "filename": self._get_filename(source),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "parsing_job_id": parsing_result.get("id"),
                "chunking_strategy": "page",
                "return_format": self.return_format,
                "pages": unique_pages,
                "elements": elements,
            }

            documents.append(Document(content=content, meta=metadata))

        return documents

    def _parse_single(self, source: Union[str, Path, ByteStream]) -> List[Document]:
        """Parse a single document."""
        try:
            # Upload file
            file_id = self._upload_file(source)

            # Create parsing job
            job = self.client.parsing.jobs.create(
                file_id=file_id,
                chunking_strategy="page",
                return_format=self.return_format,
                element_types=self.element_types,
            )

            # Wait for completion
            result = self._wait_for_completion(job.id)

            # Create documents
            return self._create_documents(result, source)

        except Exception as e:
            # Return error document
            return [
                Document(
                    content="",
                    meta={
                        "filename": self._get_filename(source),
                        "parsing_error": str(e),
                        "parsing_status": "failed",
                    },
                )
            ]

    async def _parse_single_async(
        self, source: Union[str, Path, ByteStream]
    ) -> List[Document]:
        """Parse a single document asynchronously."""
        try:
            # Upload file
            file_id = await self._upload_file_async(source)

            # Create parsing job
            job = await self.aclient.parsing.jobs.create(
                file_id=file_id,
                chunking_strategy="page",
                return_format=self.return_format,
                element_types=self.element_types,
            )

            # Wait for completion
            result = await self._wait_for_completion_async(job.id)

            # Create documents
            return self._create_documents(result, source)

        except Exception as e:
            # Return error document
            return [
                Document(
                    content="",
                    meta={
                        "filename": self._get_filename(source),
                        "parsing_error": str(e),
                        "parsing_status": "failed",
                    },
                )
            ]

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, ByteStream]]) -> Dict[str, Any]:
        """
        Parse documents from sources.

        Args:
            sources: List of file paths or ByteStream objects to parse.

        Returns:
            Dictionary containing parsed documents.
        """
        if not sources:
            return {"documents": []}

        all_documents = []
        for source in sources:
            documents = self._parse_single(source)
            all_documents.extend(documents)

        return {"documents": all_documents}

    @component.output_types(documents=List[Document])
    async def run_async(
        self, sources: List[Union[str, Path, ByteStream]]
    ) -> Dict[str, Any]:
        """
        Parse documents from sources asynchronously with concurrent processing.

        Args:
            sources: List of file paths or ByteStream objects to parse.

        Returns:
            Dictionary containing parsed documents.
        """
        if not sources:
            return {"documents": []}

        # Process files concurrently
        tasks = [self._parse_single_async(source) for source in sources]
        results = await asyncio.gather(*tasks)

        # Flatten results
        all_documents = []
        for documents in results:
            all_documents.extend(documents)

        return {"documents": all_documents}
