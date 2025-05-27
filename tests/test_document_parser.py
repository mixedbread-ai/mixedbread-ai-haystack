import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

from haystack import Document
from haystack.dataclasses import ByteStream
from haystack.utils import Secret

from mixedbread_ai_haystack.converters import MixedbreadDocumentParser


class TestMixedbreadDocumentParser:
    def test_init_default_params(self):
        """Test component initialization with default parameters."""
        parser = MixedbreadDocumentParser()

        assert parser.chunking_strategy == "page"
        assert parser.return_format == "markdown"
        assert parser.element_types == ["text", "title", "list-item", "table"]
        assert parser.max_wait_time == 300
        assert parser.poll_interval == 5
        assert not parser.store_full_path

    def test_init_custom_params(self):
        """Test component initialization with custom parameters."""
        parser = MixedbreadDocumentParser(
            chunking_strategy="paragraph",
            return_format="text",
            element_types=["text", "title"],
            max_wait_time=600,
            poll_interval=10,
            store_full_path=True,
        )

        assert parser.chunking_strategy == "paragraph"
        assert parser.return_format == "text"
        assert parser.element_types == ["text", "title"]
        assert parser.max_wait_time == 600
        assert parser.poll_interval == 10
        assert parser.store_full_path

    def test_to_dict(self):
        """Test component serialization."""
        parser = MixedbreadDocumentParser(
            api_key=Secret.from_token("test-key"),
            chunking_strategy="sentence",
            element_types=["text"],
        )

        config = parser.to_dict()

        assert (
            config["type"]
            == "mixedbread_ai_haystack.converters.document_parser.MixedbreadDocumentParser"
        )
        assert config["init_parameters"]["chunking_strategy"] == "sentence"
        assert config["init_parameters"]["element_types"] == ["text"]

    def test_from_dict(self):
        """Test component deserialization."""
        config = {
            "type": "mixedbread_ai_haystack.converters.document_parser.MixedbreadDocumentParser",
            "init_parameters": {
                "api_key": {"type": "env_var", "env_vars": ["MXBAI_API_KEY"]},
                "chunking_strategy": "paragraph",
                "return_format": "text",
                "element_types": ["text", "title"],
                "max_wait_time": 600,
                "poll_interval": 10,
                "store_full_path": True,
                "base_url": None,
                "timeout": 60.0,
                "max_retries": 2,
            },
        }

        parser = MixedbreadDocumentParser.from_dict(config)

        assert parser.chunking_strategy == "paragraph"
        assert parser.return_format == "text"
        assert parser.element_types == ["text", "title"]
        assert parser.max_wait_time == 600

    @patch(
        "mixedbread_ai_haystack.converters.document_parser.MixedbreadDocumentParser._upload_file"
    )
    @patch(
        "mixedbread_ai_haystack.converters.document_parser.MixedbreadDocumentParser._create_parsing_job"
    )
    @patch(
        "mixedbread_ai_haystack.converters.document_parser.MixedbreadDocumentParser._wait_for_job_completion"
    )
    def test_run_success(self, mock_wait, mock_create_job, mock_upload):
        """Test successful document parsing."""
        # Setup mocks
        mock_upload.return_value = "file_123"
        mock_create_job.return_value = "job_456"
        mock_wait.return_value = {
            "id": "job_456",
            "status": "completed",
            "result": {
                "chunking_strategy": "page",
                "return_format": "markdown",
                "element_types": ["text", "title"],
                "chunks": [
                    {
                        "content": "# Document Title\n\nThis is page 1 content.",
                        "content_to_embed": "Document Title This is page 1 content.",
                        "elements": [
                            {
                                "type": "title",
                                "content": "Document Title",
                                "page": 1,
                                "confidence": 0.98,
                            },
                            {
                                "type": "text",
                                "content": "This is page 1 content.",
                                "page": 1,
                                "confidence": 0.95,
                            },
                        ],
                    }
                ],
                "page_sizes": [[612, 792]],
            },
        }

        parser = MixedbreadDocumentParser()
        results = parser.run(sources=["test.pdf"])

        # Verify results
        assert "documents" in results
        documents = results["documents"]
        assert len(documents) == 1

        doc = documents[0]
        assert doc.content == "# Document Title\n\nThis is page 1 content."
        assert doc.meta["file_path"] == "test.pdf"
        assert doc.meta["parsing_job_id"] == "job_456"
        assert doc.meta["chunk_index"] == 0
        assert doc.meta["element_count"] == 2
        assert doc.meta["pages"] == [1]
        assert doc.meta["page_range"] == "1"

    def test_run_empty_sources(self):
        """Test running with empty sources."""
        parser = MixedbreadDocumentParser()
        results = parser.run(sources=[])

        assert results == {"documents": []}

    def test_run_invalid_meta_length(self):
        """Test running with mismatched meta list length."""
        parser = MixedbreadDocumentParser()

        with pytest.raises(ValueError, match="Length of meta list"):
            parser.run(sources=["file1.pdf", "file2.pdf"], meta=[{"key": "value"}])

    def test_run_invalid_meta_type(self):
        """Test running with invalid meta type."""
        parser = MixedbreadDocumentParser()

        with pytest.raises(
            TypeError, match="meta must be a dict, list of dicts, or None"
        ):
            parser.run(sources=["file1.pdf"], meta="invalid")

    @patch(
        "mixedbread_ai_haystack.converters.document_parser.MixedbreadDocumentParser._upload_file"
    )
    def test_run_with_error_handling(self, mock_upload):
        """Test error handling during parsing."""
        mock_upload.side_effect = Exception("Upload failed")

        parser = MixedbreadDocumentParser()
        results = parser.run(sources=["test.pdf"])

        documents = results["documents"]
        assert len(documents) == 1

        error_doc = documents[0]
        assert error_doc.content == ""
        assert "parsing_error" in error_doc.meta
        assert (
            "Failed to parse test.pdf: Upload failed" in error_doc.meta["parsing_error"]
        )
        assert error_doc.meta["parsing_status"] == "failed"

    def test_upload_file_with_path(self):
        """Test file upload with file path."""
        parser = MixedbreadDocumentParser()

        # Mock the client and file operations
        mock_client = Mock()
        mock_client.files.create.return_value = Mock(id="file_123")
        parser.client = mock_client

        with patch("builtins.open", mock_open(read_data=b"file content")):
            file_id = parser._upload_file("test.pdf")

        assert file_id == "file_123"
        mock_client.files.create.assert_called_once()

    def test_upload_file_with_bytestream(self):
        """Test file upload with ByteStream."""
        parser = MixedbreadDocumentParser()

        # Mock the client
        mock_client = Mock()
        mock_client.files.create.return_value = Mock(id="file_456")
        parser.client = mock_client

        bytestream = ByteStream(data=b"test content", meta={"file_path": "test.pdf"})
        file_id = parser._upload_file(bytestream)

        assert file_id == "file_456"
        mock_client.files.create.assert_called_once()

    def test_create_parsing_job(self):
        """Test parsing job creation."""
        parser = MixedbreadDocumentParser(
            chunking_strategy="paragraph",
            return_format="text",
            element_types=["text", "title"],
        )

        # Mock the client
        mock_client = Mock()
        mock_client.parsing.jobs.create.return_value = Mock(id="job_789")
        parser.client = mock_client

        job_id = parser._create_parsing_job("file_123")

        assert job_id == "job_789"
        mock_client.parsing.jobs.create.assert_called_once_with(
            file_id="file_123",
            chunking_strategy="paragraph",
            return_format="text",
            element_types=["text", "title"],
        )

    @patch("time.sleep")
    @patch("time.time")
    def test_wait_for_job_completion_success(self, mock_time, mock_sleep):
        """Test successful job completion waiting."""
        parser = MixedbreadDocumentParser()

        # Mock time progression
        mock_time.side_effect = [0, 5, 10]  # Start, first check, completion

        # Mock the client
        mock_client = Mock()
        completed_result = Mock()
        completed_result.status = "completed"
        completed_result.model_dump.return_value = {
            "id": "job_123",
            "status": "completed",
        }
        mock_client.parsing.jobs.retrieve.return_value = completed_result
        parser.client = mock_client

        result = parser._wait_for_job_completion("job_123")

        assert result == {"id": "job_123", "status": "completed"}

    @patch("time.sleep")
    @patch("time.time")
    def test_wait_for_job_completion_failed(self, mock_time, mock_sleep):
        """Test job failure during waiting."""
        parser = MixedbreadDocumentParser()

        mock_time.side_effect = [0, 5]

        # Mock the client
        mock_client = Mock()
        failed_result = Mock()
        failed_result.status = "failed"
        failed_result.error = "Processing error"
        mock_client.parsing.jobs.retrieve.return_value = failed_result
        parser.client = mock_client

        with pytest.raises(RuntimeError, match="Parsing job failed: Processing error"):
            parser._wait_for_job_completion("job_123")

    @patch("time.sleep")
    @patch("time.time")
    def test_wait_for_job_completion_timeout(self, mock_time, mock_sleep):
        """Test job timeout during waiting."""
        parser = MixedbreadDocumentParser(max_wait_time=10)

        # Mock time progression to exceed timeout
        mock_time.side_effect = [0, 5, 12]

        # Mock the client - job never completes
        mock_client = Mock()
        pending_result = Mock()
        pending_result.status = "processing"
        mock_client.parsing.jobs.retrieve.return_value = pending_result
        parser.client = mock_client

        with pytest.raises(TimeoutError, match="did not complete within 10 seconds"):
            parser._wait_for_job_completion("job_123")

    def test_create_documents_from_result_with_full_path(self):
        """Test document creation with full path storage."""
        parser = MixedbreadDocumentParser(store_full_path=True)

        parsing_result = {
            "id": "job_123",
            "result": {
                "chunking_strategy": "page",
                "return_format": "markdown",
                "element_types": ["text"],
                "chunks": [
                    {
                        "content": "Test content",
                        "content_to_embed": "Test content",
                        "elements": [
                            {"type": "text", "content": "Test content", "page": 1}
                        ],
                    }
                ],
                "page_sizes": [[612, 792]],
            },
        }

        documents = parser._create_documents_from_result(
            parsing_result, "/full/path/to/test.pdf", {"custom_meta": "value"}
        )

        assert len(documents) == 1
        doc = documents[0]
        assert doc.meta["file_path"] == "/full/path/to/test.pdf"
        assert doc.meta["custom_meta"] == "value"

    def test_create_documents_from_result_with_bytestream(self):
        """Test document creation with ByteStream source."""
        parser = MixedbreadDocumentParser(store_full_path=False)

        parsing_result = {
            "id": "job_123",
            "result": {"chunks": [{"content": "Test content", "elements": []}]},
        }

        bytestream = ByteStream(data=b"content", meta={"file_path": "stream_file.pdf"})
        documents = parser._create_documents_from_result(parsing_result, bytestream)

        assert len(documents) == 1
        doc = documents[0]
        assert doc.meta["file_path"] == "stream_file.pdf"
