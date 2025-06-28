"""
Tests for MixedbreadReranker.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock

from haystack import Document
from haystack.utils import Secret

from mixedbread_ai_haystack import MixedbreadReranker


class TestMixedbreadReranker:
    
    def test_init(self):
        """Test reranker initialization."""
        reranker = MixedbreadReranker(
            api_key=Secret.from_token("test-key"),
            model="test-model",
            top_k=5,
            return_input=True
        )
        
        assert reranker.model == "test-model"
        assert reranker.top_k == 5
        assert reranker.return_input == True
        assert reranker.timeout == 60.0
        assert reranker.max_retries == 2

    def test_init_missing_api_key(self):
        """Test initialization with missing API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="None of the following authentication environment variables are set"):
                MixedbreadReranker()

    def test_to_dict(self):
        """Test serialization to dictionary."""
        with patch.dict("os.environ", {"MXBAI_API_KEY": "test-key"}):
            reranker = MixedbreadReranker(model="test-model", top_k=5)
            
            data = reranker.to_dict()
            
            assert data["type"] == "mixedbread_ai_haystack.rerankers.reranker.MixedbreadReranker"
            assert data["init_parameters"]["model"] == "test-model"
            assert data["init_parameters"]["top_k"] == 5
            assert "api_key" in data["init_parameters"]

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "type": "mixedbread_ai_haystack.rerankers.reranker.MixedbreadReranker",
            "init_parameters": {
                "api_key": {"type": "env_var", "env_vars": ["MXBAI_API_KEY"], "strict": True},
                "model": "test-model",
                "top_k": 5
            }
        }
        
        with patch.dict("os.environ", {"MXBAI_API_KEY": "test-key"}):
            reranker = MixedbreadReranker.from_dict(data)
            assert reranker.model == "test-model"
            assert reranker.top_k == 5

    def test_prepare_texts(self):
        """Test text preparation from documents."""
        reranker = MixedbreadReranker(api_key=Secret.from_token("test-key"))
        
        documents = [
            Document(content="First document"),
            Document(content="Second document"),
            Document(content=""),  # Empty content
            Document(),  # No content
        ]
        
        texts = reranker._prepare_texts(documents)
        
        assert texts == ["First document", "Second document", "", ""]

    @patch('mixedbread_ai_haystack.rerankers.reranker.Mixedbread')
    def test_run_success(self, mock_client):
        """Test successful document reranking."""
        # Mock the client and response
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_response = Mock()
        mock_response.model = "test-model"
        mock_response.usage.model_dump.return_value = {"prompt_tokens": 10, "total_tokens": 10}
        mock_response.data = [
            Mock(index=1, score=0.95),
            Mock(index=0, score=0.85)
        ]
        
        mock_client_instance.rerank.return_value = mock_response
        
        reranker = MixedbreadReranker(api_key=Secret.from_token("test-key"))
        
        documents = [
            Document(content="First document"),
            Document(content="Second document")
        ]
        
        result = reranker.run(documents=documents, query="test query")
        
        # Check that documents are reranked
        assert len(result["documents"]) == 2
        assert result["documents"][0].content == "Second document"  # Index 1
        assert result["documents"][1].content == "First document"   # Index 0
        assert result["documents"][0].meta["rerank_score"] == 0.95
        assert result["documents"][1].meta["rerank_score"] == 0.85
        assert result["meta"]["model"] == "test-model"
        assert result["meta"]["top_k"] == 2
        
        # Verify API call
        mock_client_instance.rerank.assert_called_once_with(
            model="mixedbread-ai/mxbai-rerank-large-v2",
            query="test query",
            input=["First document", "Second document"],
            top_k=10,
            return_input=False,
        )

    @patch('mixedbread_ai_haystack.rerankers.reranker.Mixedbread')
    def test_run_empty_documents(self, mock_client):
        """Test reranking empty document list."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        reranker = MixedbreadReranker(api_key=Secret.from_token("test-key"))
        
        result = reranker.run(documents=[], query="test query")
        
        assert result["documents"] == []
        assert result["meta"]["model"] == "mixedbread-ai/mxbai-rerank-large-v2"
        assert result["meta"]["top_k"] == 0
        
        # Should not call the API for empty documents
        mock_client_instance.rerank.assert_not_called()

    @patch('mixedbread_ai_haystack.rerankers.reranker.Mixedbread')
    def test_run_empty_query(self, mock_client):
        """Test reranking with empty query."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        reranker = MixedbreadReranker(api_key=Secret.from_token("test-key"))
        
        documents = [
            Document(content="First document"),
            Document(content="Second document")
        ]
        
        result = reranker.run(documents=documents, query="")
        
        # Should return original documents without reranking
        assert len(result["documents"]) == 2
        assert result["documents"] == documents
        assert result["meta"]["top_k"] == 2
        
        # Should not call the API for empty query
        mock_client_instance.rerank.assert_not_called()

    def test_run_invalid_input(self):
        """Test reranking with invalid input type."""
        reranker = MixedbreadReranker(api_key=Secret.from_token("test-key"))
        
        with pytest.raises(TypeError, match="Input must be a list of Haystack Documents"):
            reranker.run(documents=["not", "documents"], query="test")
        
        with pytest.raises(TypeError, match="Input must be a list of Haystack Documents"):
            reranker.run(documents="not a list", query="test")

    @patch('mixedbread_ai_haystack.rerankers.reranker.Mixedbread')
    def test_run_with_custom_params(self, mock_client):
        """Test reranking with custom parameters."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_response = Mock()
        mock_response.model = "custom-model"
        mock_response.usage.model_dump.return_value = {"prompt_tokens": 5, "total_tokens": 5}
        mock_response.data = [Mock(index=0, score=0.9)]
        
        mock_client_instance.rerank.return_value = mock_response
        
        reranker = MixedbreadReranker(
            api_key=Secret.from_token("test-key"),
            model="custom-model",
            top_k=3,
            return_input=True
        )
        
        documents = [Document(content="Test document")]
        result = reranker.run(documents=documents, query="test query")
        
        mock_client_instance.rerank.assert_called_once_with(
            model="custom-model",
            query="test query",
            input=["Test document"],
            top_k=3,
            return_input=True,
        )

    @pytest.mark.asyncio
    @patch('mixedbread_ai_haystack.rerankers.reranker.AsyncMixedbread')
    async def test_run_async_success(self, mock_async_client):
        """Test successful async document reranking."""
        # Mock the async client and response
        mock_client_instance = AsyncMock()
        mock_async_client.return_value = mock_client_instance
        
        mock_response = Mock()
        mock_response.model = "test-model"
        mock_response.usage.model_dump.return_value = {"prompt_tokens": 10, "total_tokens": 10}
        mock_response.data = [
            Mock(index=1, score=0.9),
            Mock(index=0, score=0.8)
        ]
        
        mock_client_instance.rerank.return_value = mock_response
        
        reranker = MixedbreadReranker(api_key=Secret.from_token("test-key"))
        
        documents = [
            Document(content="Async document 1"),
            Document(content="Async document 2")
        ]
        
        result = await reranker.run_async(documents=documents, query="async test")
        
        # Check that documents are reranked
        assert len(result["documents"]) == 2
        assert result["documents"][0].content == "Async document 2"  # Index 1
        assert result["documents"][1].content == "Async document 1"  # Index 0
        assert result["documents"][0].meta["rerank_score"] == 0.9
        assert result["documents"][1].meta["rerank_score"] == 0.8
        assert result["meta"]["model"] == "test-model"
        
        # Verify async API call
        mock_client_instance.rerank.assert_called_once_with(
            model="mixedbread-ai/mxbai-rerank-large-v2",
            query="async test",
            input=["Async document 1", "Async document 2"],
            top_k=10,
            return_input=False,
        )

    @pytest.mark.asyncio
    @patch('mixedbread_ai_haystack.rerankers.reranker.AsyncMixedbread')
    async def test_run_async_empty_documents(self, mock_async_client):
        """Test async reranking with empty document list."""
        mock_client_instance = AsyncMock()
        mock_async_client.return_value = mock_client_instance
        
        reranker = MixedbreadReranker(api_key=Secret.from_token("test-key"))
        
        result = await reranker.run_async(documents=[], query="test")
        
        assert result["documents"] == []
        assert result["meta"]["top_k"] == 0
        
        # Should not call the API for empty documents
        mock_client_instance.rerank.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_async_invalid_input(self):
        """Test async reranking with invalid input type."""
        reranker = MixedbreadReranker(api_key=Secret.from_token("test-key"))
        
        with pytest.raises(TypeError, match="Input must be a list of Haystack Documents"):
            await reranker.run_async(documents=["not", "documents"], query="test")

    def test_create_metadata(self):
        """Test metadata creation from response."""
        reranker = MixedbreadReranker(api_key=Secret.from_token("test-key"))
        
        mock_response = Mock()
        mock_response.model = "test-model"
        mock_response.usage.model_dump.return_value = {"prompt_tokens": 15, "total_tokens": 15}
        
        meta = reranker._create_metadata(mock_response, 3)
        
        assert meta["model"] == "test-model"
        assert meta["usage"]["prompt_tokens"] == 15
        assert meta["top_k"] == 3

    @patch('mixedbread_ai_haystack.rerankers.reranker.Mixedbread')
    def test_document_metadata_preservation(self, mock_client):
        """Test that original document metadata is preserved."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_response = Mock()
        mock_response.model = "test-model"
        mock_response.usage.model_dump.return_value = {"prompt_tokens": 5, "total_tokens": 5}
        mock_response.data = [Mock(index=0, score=0.85)]
        
        mock_client_instance.rerank.return_value = mock_response
        
        reranker = MixedbreadReranker(api_key=Secret.from_token("test-key"))
        
        original_doc = Document(content="Original content", meta={"key": "value"})
        documents = [original_doc]
        
        result = reranker.run(documents=documents, query="test")
        
        # Check that original content and metadata are preserved
        reranked_doc = result["documents"][0]
        assert reranked_doc.content == "Original content"
        assert reranked_doc.meta["key"] == "value"
        assert reranked_doc.meta["rerank_score"] == 0.85
        
        # Ensure it's the same document object (modified in place)
        assert reranked_doc is original_doc