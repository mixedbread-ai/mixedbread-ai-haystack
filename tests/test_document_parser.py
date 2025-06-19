"""
Tests for MixedbreadDocumentParser.
"""
import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from haystack import Document
from haystack.dataclasses import ByteStream
from haystack.utils import Secret

from mixedbread_ai_haystack import MixedbreadDocumentParser


class TestMixedbreadDocumentParser:
    
    def test_init(self):
        """Test parser initialization."""
        parser = MixedbreadDocumentParser(
            api_key=Secret.from_token("test-key"),
            return_format="plain"
        )
        
        assert parser.return_format == "plain"
        assert parser.element_types == ["text", "title", "list-item", "table", "section-header"]
        assert parser.max_wait_time == 300
        assert parser.poll_interval == 5

    def test_init_missing_api_key(self):
        """Test initialization with missing API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="None of the following authentication environment variables are set"):
                MixedbreadDocumentParser()

    def test_to_dict(self):
        """Test serialization to dictionary."""
        with patch.dict("os.environ", {"MXBAI_API_KEY": "test-key"}):
            parser = MixedbreadDocumentParser()
            
            data = parser.to_dict()
            
            assert data["type"] == "mixedbread_ai_haystack.converters.document_parser.MixedbreadDocumentParser"
            assert data["init_parameters"]["return_format"] == "markdown"
            assert "api_key" in data["init_parameters"]

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "type": "mixedbread_ai_haystack.converters.document_parser.MixedbreadDocumentParser",
            "init_parameters": {
                "api_key": {"type": "env_var", "env_vars": ["MXBAI_API_KEY"], "strict": True},
                "return_format": "markdown"
            }
        }
        
        with patch.dict("os.environ", {"MXBAI_API_KEY": "test-key"}):
            parser = MixedbreadDocumentParser.from_dict(data)
            assert parser.return_format == "markdown"

    def test_get_filename(self):
        """Test filename extraction."""
        parser = MixedbreadDocumentParser(api_key=Secret.from_token("test-key"))
        
        # Test with string path
        assert parser._get_filename("test.pdf") == "test.pdf"
        
        # Test with Path object
        assert parser._get_filename(Path("docs/test.pdf")) == "test.pdf"
        
        # Test with ByteStream
        stream = ByteStream(data=b"test", meta={"file_path": "stream.pdf"})
        assert parser._get_filename(stream) == "stream.pdf"

    @patch('mixedbread_ai_haystack.converters.document_parser.Mixedbread')
    def test_run_empty_sources(self, mock_client):
        """Test run with empty sources."""
        parser = MixedbreadDocumentParser(api_key=Secret.from_token("test-key"))
        
        result = parser.run(sources=[])
        
        assert result == {"documents": []}
        mock_client.assert_called_once()

    @patch('mixedbread_ai_haystack.converters.document_parser.Mixedbread')
    def test_run_single_document_success(self, mock_client):
        """Test successful parsing of single document."""
        # Mock the client and its methods
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        # Mock file upload
        mock_client_instance.files.create.return_value = Mock(id="file-123")
        
        # Mock job creation
        mock_client_instance.parsing.jobs.create.return_value = Mock(id="job-456")
        
        # Mock job completion
        mock_job_result = Mock()
        mock_job_result.status = "completed"
        mock_job_result.model_dump.return_value = {
            "id": "job-456",
            "result": {
                "chunks": [
                    {
                        "content": "Test content",
                        "elements": [{"type": "text", "page": 1}]
                    }
                ]
            }
        }
        mock_client_instance.parsing.jobs.retrieve.return_value = mock_job_result
        
        parser = MixedbreadDocumentParser(api_key=Secret.from_token("test-key"))
        
        with patch("builtins.open", mock_open_function()):
            result = parser.run(sources=["test.pdf"])
        
        documents = result["documents"]
        assert len(documents) == 1
        
        doc = documents[0]
        assert doc.content == "Test content"
        assert doc.meta["filename"] == "test.pdf"
        assert doc.meta["parsing_job_id"] == "job-456"
        assert doc.meta["pages"] == [1]

    @patch('mixedbread_ai_haystack.converters.document_parser.Mixedbread')
    def test_run_parsing_error(self, mock_client):
        """Test handling of parsing errors."""
        # Mock the client to raise an exception
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.files.create.side_effect = Exception("Upload failed")
        
        parser = MixedbreadDocumentParser(api_key=Secret.from_token("test-key"))
        
        with patch("builtins.open", mock_open_function()):
            result = parser.run(sources=["test.pdf"])
        
        documents = result["documents"]
        assert len(documents) == 1
        
        doc = documents[0]
        assert doc.content == ""
        assert doc.meta["parsing_status"] == "failed"
        assert "Upload failed" in doc.meta["parsing_error"]

    @pytest.mark.asyncio
    @patch('mixedbread_ai_haystack.converters.document_parser.AsyncMixedbread')
    async def test_run_async_success(self, mock_async_client):
        """Test successful async parsing."""
        # Mock the async client
        mock_client_instance = AsyncMock()
        mock_async_client.return_value = mock_client_instance
        
        # Mock async file upload
        mock_client_instance.files.create.return_value = Mock(id="file-123")
        
        # Mock async job creation
        mock_client_instance.parsing.jobs.create.return_value = Mock(id="job-456")
        
        # Mock async job completion
        mock_job_result = Mock()
        mock_job_result.status = "completed"
        mock_job_result.model_dump.return_value = {
            "id": "job-456",
            "result": {
                "chunks": [
                    {
                        "content": "Async content",
                        "elements": [{"type": "text", "page": 1}]
                    }
                ]
            }
        }
        mock_client_instance.parsing.jobs.retrieve.return_value = mock_job_result
        
        parser = MixedbreadDocumentParser(api_key=Secret.from_token("test-key"))
        
        with patch("builtins.open", mock_open_function()):
            result = await parser.run_async(sources=["test.pdf"])
        
        documents = result["documents"]
        assert len(documents) == 1
        
        doc = documents[0]
        assert doc.content == "Async content"
        assert doc.meta["filename"] == "test.pdf"

    @pytest.mark.asyncio
    @patch('mixedbread_ai_haystack.converters.document_parser.AsyncMixedbread')
    async def test_run_async_multiple_files(self, mock_async_client):
        """Test async parsing with multiple files."""
        # Mock the async client
        mock_client_instance = AsyncMock()
        mock_async_client.return_value = mock_client_instance
        
        # Mock responses for multiple files
        mock_client_instance.files.create.side_effect = [
            Mock(id="file-1"), Mock(id="file-2")
        ]
        mock_client_instance.parsing.jobs.create.side_effect = [
            Mock(id="job-1"), Mock(id="job-2")
        ]
        
        mock_job_result = Mock()
        mock_job_result.status = "completed"
        mock_job_result.model_dump.return_value = {
            "id": "job-1",
            "result": {
                "chunks": [{"content": "Content", "elements": []}]
            }
        }
        mock_client_instance.parsing.jobs.retrieve.return_value = mock_job_result
        
        parser = MixedbreadDocumentParser(api_key=Secret.from_token("test-key"))
        
        with patch("builtins.open", mock_open_function()):
            result = await parser.run_async(sources=["test1.pdf", "test2.pdf"])
        
        documents = result["documents"]
        assert len(documents) == 2  # One chunk per file

    def test_create_documents(self):
        """Test document creation from parsing result."""
        parser = MixedbreadDocumentParser(api_key=Secret.from_token("test-key"))
        
        parsing_result = {
            "id": "job-123",
            "result": {
                "chunks": [
                    {
                        "content": "First chunk",
                        "elements": [{"type": "text", "page": 1}]
                    },
                    {
                        "content": "Second chunk", 
                        "elements": [{"type": "text", "page": 2}]
                    }
                ]
            }
        }
        
        documents = parser._create_documents(parsing_result, "test.pdf")
        
        assert len(documents) == 2
        
        # Check first document
        doc1 = documents[0]
        assert doc1.content == "First chunk"
        assert doc1.meta["filename"] == "test.pdf"
        assert doc1.meta["chunk_index"] == 0
        assert doc1.meta["total_chunks"] == 2
        assert doc1.meta["pages"] == [1]
        
        # Check second document
        doc2 = documents[1]
        assert doc2.content == "Second chunk"
        assert doc2.meta["chunk_index"] == 1
        assert doc2.meta["pages"] == [2]


def mock_open_function():
    """Create a mock open function for file operations."""
    from unittest.mock import mock_open
    return mock_open(read_data=b"test file content")