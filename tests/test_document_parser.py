import pytest
from unittest.mock import Mock, patch, mock_open, PropertyMock
from haystack.dataclasses import ByteStream
from haystack.utils import Secret
from haystack import Document

from mixedbread_ai_haystack.converters import MixedbreadDocumentParser
from tests.test_config import TestConfig


class TestMixedbreadDocumentParser:
    def test_init_default_params(self):
        """Test component initialization with default parameters."""
        parser = MixedbreadDocumentParser()

        assert parser.chunking_strategy == "page"
        assert parser.return_format == "markdown"
        assert parser.element_types == ["text", "title", "list-item", "table"]
        assert parser.max_wait_time == 300
        assert parser.poll_interval == 5

    def test_init_custom_params(self):
        """Test component initialization with custom parameters."""
        parser = MixedbreadDocumentParser(
            chunking_strategy="page",
            return_format="text",
            element_types=["text", "title"],
            max_wait_time=600,
            poll_interval=10,
        )

        assert parser.chunking_strategy == "page"
        assert parser.return_format == "text"
        assert parser.element_types == ["text", "title"]
        assert parser.max_wait_time == 600
        assert parser.poll_interval == 10

    def test_to_dict(self):
        """Test component serialization."""
        parser = MixedbreadDocumentParser(
            api_key=Secret.from_env_var("MXBAI_API_KEY"),
            chunking_strategy="page",
            element_types=["text"],
        )

        config = parser.to_dict()

        assert (
            config["type"]
            == "mixedbread_ai_haystack.converters.document_parser.MixedbreadDocumentParser"
        )
        assert config["init_parameters"]["chunking_strategy"] == "page"
        assert config["init_parameters"]["element_types"] == ["text"]

    def test_from_dict(self):
        """Test component deserialization."""
        config = {
            "type": "mixedbread_ai_haystack.converters.document_parser.MixedbreadDocumentParser",
            "init_parameters": {
                "api_key": {
                    "type": "env_var",
                    "env_vars": ["MXBAI_API_KEY"],
                    "strict": True,
                },
                "chunking_strategy": "page",
                "return_format": "text",
                "element_types": ["text", "title"],
                "max_wait_time": 600,
                "poll_interval": 10,
                "base_url": None,
                "timeout": 60.0,
                "max_retries": 2,
            },
        }

        parser = MixedbreadDocumentParser.from_dict(config)

        assert parser.chunking_strategy == "page"
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
        """Test run with invalid meta type."""
        parser = MixedbreadDocumentParser()

        with pytest.raises(ValueError):
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

    @patch(
        "mixedbread_ai_haystack.converters.document_parser.MixedbreadDocumentParser.client",
        new_callable=PropertyMock,
    )
    def test_upload_file_with_path(self, mock_client_property):
        """Test file upload with file path."""
        parser = MixedbreadDocumentParser()

        mock_client = Mock()
        mock_client.files.create.return_value = Mock(id="file_123")
        mock_client_property.return_value = mock_client

        with patch("builtins.open", mock_open(read_data=b"file content")):
            file_id = parser._upload_file("test.pdf")

        assert file_id == "file_123"
        mock_client.files.create.assert_called_once()

    @patch(
        "mixedbread_ai_haystack.converters.document_parser.MixedbreadDocumentParser.client",
        new_callable=PropertyMock,
    )
    def test_upload_file_with_bytestream(self, mock_client_property):
        """Test file upload with ByteStream."""
        parser = MixedbreadDocumentParser()

        mock_client = Mock()
        mock_client.files.create.return_value = Mock(id="file_456")
        mock_client_property.return_value = mock_client

        bytestream = ByteStream(data=b"test content", meta={"file_path": "test.pdf"})
        file_id = parser._upload_file(bytestream)

        assert file_id == "file_456"
        mock_client.files.create.assert_called_once()

    @patch(
        "mixedbread_ai_haystack.converters.document_parser.MixedbreadDocumentParser.client",
        new_callable=PropertyMock,
    )
    @patch("time.sleep")
    @patch("time.time")
    def test_wait_for_job_completion_success(
        self, mock_time, mock_sleep, mock_client_property
    ):
        """Test successful job completion waiting."""
        parser = MixedbreadDocumentParser()

        mock_time.side_effect = [0, 5, 10]

        mock_client = Mock()
        completed_result = Mock()
        completed_result.status = "completed"
        completed_result.model_dump.return_value = {
            "id": "job_123",
            "status": "completed",
        }
        mock_client.parsing.jobs.retrieve.return_value = completed_result
        mock_client_property.return_value = mock_client

        result = parser._wait_for_job_completion("job_123")

        assert result == {"id": "job_123", "status": "completed"}

    @patch(
        "mixedbread_ai_haystack.converters.document_parser.MixedbreadDocumentParser.client",
        new_callable=PropertyMock,
    )
    @patch("time.sleep")
    @patch("time.time")
    def test_wait_for_job_completion_failed(
        self, mock_time, mock_sleep, mock_client_property
    ):
        """Test job failure during waiting."""
        parser = MixedbreadDocumentParser()

        mock_time.side_effect = [0, 5]

        # Mock the client
        mock_client = Mock()
        failed_result = Mock()
        failed_result.status = "failed"
        failed_result.error = "Processing error"
        mock_client.parsing.jobs.retrieve.return_value = failed_result
        mock_client_property.return_value = mock_client

        with pytest.raises(RuntimeError, match="Parsing job failed: Processing error"):
            parser._wait_for_job_completion("job_123")

    @patch(
        "mixedbread_ai_haystack.converters.document_parser.MixedbreadDocumentParser.client",
        new_callable=PropertyMock,
    )
    @patch("time.sleep")
    @patch("time.time")
    def test_wait_for_job_completion_timeout(
        self, mock_time, mock_sleep, mock_client_property
    ):
        """Test job timeout during waiting."""
        parser = MixedbreadDocumentParser(max_wait_time=10)

        mock_time.side_effect = [0, 5, 12]

        mock_client = Mock()
        pending_result = Mock()
        pending_result.status = "processing"
        mock_client.parsing.jobs.retrieve.return_value = pending_result
        mock_client_property.return_value = mock_client

        with pytest.raises(TimeoutError, match="did not complete within 10 seconds"):
            parser._wait_for_job_completion("job_123")

    def test_create_documents_from_result_with_path(self):
        """Test document creation with file path."""
        parser = MixedbreadDocumentParser()

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
        assert doc.meta["file_path"] == "test.pdf"  # Only filename, not full path
        assert doc.meta["custom_meta"] == "value"

    def test_create_documents_from_result_with_bytestream(self):
        """Test document creation with ByteStream source."""
        parser = MixedbreadDocumentParser()

        parsing_result = {
            "id": "job_123",
            "result": {"chunks": [{"content": "Test content", "elements": []}]},
        }

        bytestream = ByteStream(data=b"content", meta={"file_path": "stream_file.pdf"})
        documents = parser._create_documents_from_result(parsing_result, bytestream)

        assert len(documents) == 1
        doc = documents[0]
        assert doc.meta["file_path"] == "stream_file.pdf"

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    def test_integration_basic_parsing(self):
        """
        Test basic document parsing with real API call.
        
        Uses the example PDF file included in the repository for testing.
        """
        import os
        from pathlib import Path

        test_file_path = Path(__file__).parent.parent / "data" / "acme_invoice.pdf"

        if not test_file_path.exists():
            pytest.skip(f"Test file not found: {test_file_path}")

        parser_config = TestConfig.get_test_parser_config()
        parser = MixedbreadDocumentParser(**parser_config)

        result = parser.run(sources=[str(test_file_path)])
        documents = result["documents"]

        assert isinstance(documents, list)
        assert len(documents) > 0

        for doc in documents:
            assert isinstance(doc, Document)
            assert isinstance(doc.content, str)
            assert len(doc.content) > 0

            assert "file_path" in doc.meta
            assert "parsing_job_id" in doc.meta
            assert "chunking_strategy" in doc.meta
            assert "return_format" in doc.meta
            assert "element_types" in doc.meta
            assert "chunk_index" in doc.meta
            assert isinstance(doc.meta["chunk_index"], int)

            assert doc.meta.get("parsing_status") != "failed"
            assert "parsing_error" not in doc.meta

        has_content = any(len(doc.content.strip()) > 0 for doc in documents)
        assert has_content, "No documents contain actual content"
