"""Tests for MixedbreadVectorStoreRetriever."""
import os
import pytest
from unittest.mock import AsyncMock, Mock, patch
from haystack import Document
from mixedbread_ai_haystack import MixedbreadVectorStoreRetriever


class TestMixedbreadVectorStoreRetriever:
    """Test MixedbreadVectorStoreRetriever functionality."""

    def test_init_with_defaults(self):
        """Test retriever initialization with default parameters."""
        with patch.dict(os.environ, {"MXBAI_API_KEY": "test-key"}):
            retriever = MixedbreadVectorStoreRetriever(
                vector_store_identifiers=["store-1"]
            )
            
            assert retriever.vector_store_identifiers == ["store-1"]
            assert retriever.search_type == "chunk"
            assert retriever.top_k == 10
            assert retriever.score_threshold is None

    def test_init_with_custom_params(self):
        """Test retriever initialization with custom parameters."""
        with patch.dict(os.environ, {"MXBAI_API_KEY": "test-key"}):
            retriever = MixedbreadVectorStoreRetriever(
                vector_store_identifiers=["store-1", "store-2"],
                search_type="file",
                top_k=5,
                score_threshold=0.8
            )
            
            assert retriever.vector_store_identifiers == ["store-1", "store-2"]
            assert retriever.search_type == "file"
            assert retriever.top_k == 5
            assert retriever.score_threshold == 0.8

    def test_init_missing_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict(os.environ, {"MXBAI_API_KEY": ""}):
            with pytest.raises(ValueError, match="Mixedbread API key not found"):
                MixedbreadVectorStoreRetriever(vector_store_identifiers=["store-1"])

    def test_init_empty_vector_stores(self):
        """Test initialization fails with empty vector store list."""
        with patch.dict(os.environ, {"MXBAI_API_KEY": "test-key"}):
            with pytest.raises(ValueError, match="At least one vector store identifier must be provided"):
                MixedbreadVectorStoreRetriever(vector_store_identifiers=[])

    def test_to_dict_from_dict(self):
        """Test serialization and deserialization."""
        with patch.dict(os.environ, {"MXBAI_API_KEY": "test-key"}):
            retriever = MixedbreadVectorStoreRetriever(
                vector_store_identifiers=["store-1"],
                search_type="file",
                top_k=5
            )
            
            # Serialize
            data = retriever.to_dict()
            assert data["init_parameters"]["vector_store_identifiers"] == ["store-1"]
            assert data["init_parameters"]["search_type"] == "file"
            assert data["init_parameters"]["top_k"] == 5
            
            # Deserialize
            new_retriever = MixedbreadVectorStoreRetriever.from_dict(data)
            assert new_retriever.vector_store_identifiers == ["store-1"]
            assert new_retriever.search_type == "file"
            assert new_retriever.top_k == 5

    @patch("mixedbread_ai_haystack.retrievers.vector_store_retriever.Mixedbread")
    def test_run_chunk_search(self, mock_mixedbread):
        """Test chunk search functionality."""
        # Mock API response with proper object structure
        mock_chunk1 = Mock()
        mock_chunk1.text = "Test content 1"
        mock_chunk1.type = "text"
        mock_chunk1.score = 0.95
        mock_chunk1.chunk_index = 0
        mock_chunk1.filename = "test.pdf"
        mock_chunk1.metadata = {}
        
        mock_chunk2 = Mock()
        mock_chunk2.text = "Test content 2"
        mock_chunk2.type = "text"
        mock_chunk2.score = 0.85
        mock_chunk2.chunk_index = 1
        mock_chunk2.filename = "test.pdf"
        mock_chunk2.metadata = {}
        
        mock_response = Mock()
        mock_response.data = [mock_chunk1, mock_chunk2]
        
        mock_client = Mock()
        mock_client.vector_stores.search.return_value = mock_response
        mock_mixedbread.return_value = mock_client
        
        with patch.dict(os.environ, {"MXBAI_API_KEY": "test-key"}):
            retriever = MixedbreadVectorStoreRetriever(
                vector_store_identifiers=["store-1"],
                search_type="chunk"
            )
            
            result = retriever.run(query="test query")
            
            # Verify API call
            mock_client.vector_stores.search.assert_called_once_with(
                query="test query",
                vector_store_identifiers=["store-1"],
                top_k=10,
                search_options={"return_metadata": True}
            )
            
            # Verify results
            assert len(result["documents"]) == 2
            assert result["documents"][0].content == "Test content 1"
            assert result["documents"][0].meta["retrieval_score"] == 0.95
            assert result["documents"][0].meta["chunk_index"] == 0
            assert result["documents"][0].meta["filename"] == "test.pdf"
            assert result["meta"]["search_type"] == "chunk"

    @patch("mixedbread_ai_haystack.retrievers.vector_store_retriever.Mixedbread")
    def test_run_file_search(self, mock_mixedbread):
        """Test file search functionality."""
        # Mock chunk for file
        mock_chunk = Mock()
        mock_chunk.text = "Full file content"
        mock_chunk.type = "text"
        
        # Mock file response
        mock_file = Mock()
        mock_file.score = 0.9
        mock_file.id = "file123"
        mock_file.filename = "document.pdf"
        mock_file.chunks = [mock_chunk]
        mock_file.metadata = {}
        
        mock_response = Mock()
        mock_response.data = [mock_file]
        
        mock_client = Mock()
        mock_client.vector_stores.files.search.return_value = mock_response
        mock_mixedbread.return_value = mock_client
        
        with patch.dict(os.environ, {"MXBAI_API_KEY": "test-key"}):
            retriever = MixedbreadVectorStoreRetriever(
                vector_store_identifiers=["store-1"],
                search_type="file"
            )
            
            result = retriever.run(query="test query", top_k=3)
            
            # Verify API call
            mock_client.vector_stores.files.search.assert_called_once_with(
                query="test query",
                vector_store_identifiers=["store-1"],
                top_k=3,
                search_options={"return_metadata": True, "return_chunks": True}
            )
            
            # Verify results
            assert len(result["documents"]) == 1
            assert result["documents"][0].content == "Full file content"
            assert result["documents"][0].meta["retrieval_score"] == 0.9
            assert result["documents"][0].meta["file_id"] == "file123"
            assert result["meta"]["search_type"] == "file"

    @patch("mixedbread_ai_haystack.retrievers.vector_store_retriever.Mixedbread")
    def test_run_with_score_threshold(self, mock_mixedbread):
        """Test filtering by score threshold."""
        # Mock chunks with varying scores
        mock_chunk1 = Mock()
        mock_chunk1.text = "High score"
        mock_chunk1.type = "text"
        mock_chunk1.score = 0.9
        mock_chunk1.metadata = {}
        
        mock_chunk2 = Mock()
        mock_chunk2.text = "Medium score"
        mock_chunk2.type = "text"
        mock_chunk2.score = 0.7
        mock_chunk2.metadata = {}
        
        mock_chunk3 = Mock()
        mock_chunk3.text = "Low score"
        mock_chunk3.type = "text"
        mock_chunk3.score = 0.5
        mock_chunk3.metadata = {}
        
        mock_response = Mock()
        mock_response.data = [mock_chunk1, mock_chunk2, mock_chunk3]
        
        mock_client = Mock()
        mock_client.vector_stores.search.return_value = mock_response
        mock_mixedbread.return_value = mock_client
        
        with patch.dict(os.environ, {"MXBAI_API_KEY": "test-key"}):
            retriever = MixedbreadVectorStoreRetriever(
                vector_store_identifiers=["store-1"],
                score_threshold=0.8
            )
            
            result = retriever.run(query="test query")
            
            # Client-side filtering should only return results with score >= 0.8
            assert len(result["documents"]) == 1  # Only high score meets threshold
            assert result["documents"][0].content == "High score"
            assert result["documents"][0].meta["retrieval_score"] == 0.9

    @patch("mixedbread_ai_haystack.retrievers.vector_store_retriever.Mixedbread")
    def test_run_empty_query(self, mock_mixedbread):
        """Test handling of empty query."""
        with patch.dict(os.environ, {"MXBAI_API_KEY": "test-key"}):
            retriever = MixedbreadVectorStoreRetriever(vector_store_identifiers=["store-1"])
            
            result = retriever.run(query="")
            
            assert result["documents"] == []

    @patch("mixedbread_ai_haystack.retrievers.vector_store_retriever.Mixedbread")
    def test_run_api_error(self, mock_mixedbread):
        """Test handling of API errors."""
        mock_client = Mock()
        mock_client.vector_stores.search.side_effect = Exception("API Error")
        mock_mixedbread.return_value = mock_client
        
        with patch.dict(os.environ, {"MXBAI_API_KEY": "test-key"}):
            retriever = MixedbreadVectorStoreRetriever(vector_store_identifiers=["store-1"])
            
            result = retriever.run(query="test query")
            
            assert result["documents"] == []
            assert "error" in result["meta"]
            assert result["meta"]["error"] == "API Error"

    @patch("mixedbread_ai_haystack.retrievers.vector_store_retriever.AsyncMixedbread")
    @pytest.mark.asyncio
    async def test_run_async(self, mock_async_mixedbread):
        """Test async search functionality."""
        # Mock async chunk
        mock_chunk = Mock()
        mock_chunk.text = "Async content"
        mock_chunk.type = "text"
        mock_chunk.score = 0.95
        mock_chunk.metadata = {}
        
        mock_response = Mock()
        mock_response.data = [mock_chunk]
        
        mock_aclient = AsyncMock()
        mock_aclient.vector_stores.search.return_value = mock_response
        mock_async_mixedbread.return_value = mock_aclient
        
        with patch.dict(os.environ, {"MXBAI_API_KEY": "test-key"}):
            retriever = MixedbreadVectorStoreRetriever(
                vector_store_identifiers=["store-1", "store-2"]
            )
            
            result = await retriever.run_async(query="async test")
            
            # Verify async API call
            mock_aclient.vector_stores.search.assert_called_once_with(
                query="async test",
                vector_store_identifiers=["store-1", "store-2"],
                top_k=10,
                search_options={"return_metadata": True}
            )
            
            # Verify results
            assert len(result["documents"]) == 1
            assert result["documents"][0].content == "Async content"
            assert result["meta"]["vector_stores"] == ["store-1", "store-2"]

    def test_extract_content_variations(self):
        """Test content extraction from different formats."""
        with patch.dict(os.environ, {"MXBAI_API_KEY": "test-key"}):
            retriever = MixedbreadVectorStoreRetriever(vector_store_identifiers=["store-1"])
            
            # Text chunk
            text_chunk = Mock()
            text_chunk.type = "text"
            text_chunk.text = "Text content"
            assert retriever._extract_content(text_chunk) == "Text content"
            
            # Image chunk with OCR
            image_chunk = Mock()
            image_chunk.type = "image_url"
            image_chunk.ocr_text = "OCR text"
            assert retriever._extract_content(image_chunk) == "OCR text"
            
            # Object without content
            empty_obj = Mock()
            del empty_obj.type  # Remove type attribute
            del empty_obj.text  # Remove text attribute
            assert retriever._extract_content(empty_obj) == ""

    def test_create_metadata(self):
        """Test metadata creation."""
        with patch.dict(os.environ, {"MXBAI_API_KEY": "test-key"}):
            retriever = MixedbreadVectorStoreRetriever(vector_store_identifiers=["store-1"])
            
            # Mock chunk object
            chunk = Mock()
            chunk.score = 0.85
            chunk.chunk_index = 2
            chunk.filename = "test.pdf"
            chunk.file_id = "file123"
            chunk.metadata = {"custom": "value"}
            # Remove id attribute to avoid conflict
            if hasattr(chunk, 'id'):
                delattr(chunk, 'id')
            
            metadata = retriever._create_metadata(chunk, "chunk")
            
            assert metadata["retrieval_score"] == 0.85
            assert metadata["search_type"] == "chunk"
            assert metadata["chunk_index"] == 2
            assert metadata["filename"] == "test.pdf"
            assert metadata["file_id"] == "file123"
            assert metadata["custom"] == "value"