"""
Tests for MixedbreadDocumentEmbedder.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock

from haystack import Document
from haystack.utils import Secret

from mixedbread_ai_haystack import MixedbreadDocumentEmbedder


class TestMixedbreadDocumentEmbedder:
    
    def test_init(self):
        """Test embedder initialization."""
        embedder = MixedbreadDocumentEmbedder(
            api_key=Secret.from_token("test-key"),
            model="test-model",
            normalized=False
        )
        
        assert embedder.model == "test-model"
        assert embedder.normalized == False
        assert embedder.encoding_format == "float"
        assert embedder.dimensions is None
        assert embedder.timeout == 60.0

    def test_init_missing_api_key(self):
        """Test initialization with missing API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="None of the following authentication environment variables are set"):
                MixedbreadDocumentEmbedder()

    def test_to_dict(self):
        """Test serialization to dictionary."""
        with patch.dict("os.environ", {"MXBAI_API_KEY": "test-key"}):
            embedder = MixedbreadDocumentEmbedder(model="test-model")
            
            data = embedder.to_dict()
            
            assert data["type"] == "mixedbread_ai_haystack.embedders.document_embedder.MixedbreadDocumentEmbedder"
            assert data["init_parameters"]["model"] == "test-model"
            assert "api_key" in data["init_parameters"]

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "type": "mixedbread_ai_haystack.embedders.document_embedder.MixedbreadDocumentEmbedder",
            "init_parameters": {
                "api_key": {"type": "env_var", "env_vars": ["MXBAI_API_KEY"], "strict": True},
                "model": "test-model",
                "normalized": False
            }
        }
        
        with patch.dict("os.environ", {"MXBAI_API_KEY": "test-key"}):
            embedder = MixedbreadDocumentEmbedder.from_dict(data)
            assert embedder.model == "test-model"
            assert embedder.normalized == False

    def test_prepare_texts(self):
        """Test text preparation from documents."""
        embedder = MixedbreadDocumentEmbedder(api_key=Secret.from_token("test-key"))
        
        documents = [
            Document(content="First document"),
            Document(content="Second document"),
            Document(content=""),  # Empty content
            Document(),  # No content
        ]
        
        texts = embedder._prepare_texts(documents)
        
        assert texts == ["First document", "Second document", "", ""]

    @patch('mixedbread_ai_haystack.embedders.document_embedder.Mixedbread')
    def test_run_success(self, mock_client):
        """Test successful document embedding."""
        # Mock the client and response
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_response = Mock()
        mock_response.model = "test-model"
        mock_response.usage.model_dump.return_value = {"prompt_tokens": 10, "total_tokens": 10}
        mock_response.normalized = True
        mock_response.encoding_format = "float"
        mock_response.dimensions = 1024
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        
        mock_client_instance.embed.return_value = mock_response
        
        embedder = MixedbreadDocumentEmbedder(api_key=Secret.from_token("test-key"))
        
        documents = [
            Document(content="First document"),
            Document(content="Second document")
        ]
        
        result = embedder.run(documents)
        
        # Check that embeddings are attached to documents
        assert len(result["documents"]) == 2
        assert result["documents"][0].embedding == [0.1, 0.2, 0.3]
        assert result["documents"][1].embedding == [0.4, 0.5, 0.6]
        assert result["meta"]["model"] == "test-model"
        assert result["meta"]["normalized"] == True
        
        # Verify API call
        mock_client_instance.embed.assert_called_once_with(
            model="mixedbread-ai/mxbai-embed-large-v1",
            input=["First document", "Second document"],
            normalized=True,
            encoding_format="float",
            dimensions=None,
            prompt=None,
        )

    @patch('mixedbread_ai_haystack.embedders.document_embedder.Mixedbread')
    def test_run_empty_documents(self, mock_client):
        """Test embedding empty document list."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        embedder = MixedbreadDocumentEmbedder(api_key=Secret.from_token("test-key"))
        
        result = embedder.run([])
        
        assert result["documents"] == []
        assert result["meta"]["model"] == "mixedbread-ai/mxbai-embed-large-v1"
        assert result["meta"]["dimensions"] == 0
        
        # Should not call the API for empty documents
        mock_client_instance.embed.assert_not_called()

    def test_run_invalid_input(self):
        """Test embedding with invalid input type."""
        embedder = MixedbreadDocumentEmbedder(api_key=Secret.from_token("test-key"))
        
        with pytest.raises(TypeError, match="Input must be a list of Haystack Documents"):
            embedder.run(["not", "documents"])
        
        with pytest.raises(TypeError, match="Input must be a list of Haystack Documents"):
            embedder.run("not a list")

    @patch('mixedbread_ai_haystack.embedders.document_embedder.Mixedbread')
    def test_run_with_prompt(self, mock_client):
        """Test embedding with custom prompt."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_response = Mock()
        mock_response.model = "test-model"
        mock_response.usage.model_dump.return_value = {"prompt_tokens": 5, "total_tokens": 5}
        mock_response.normalized = True
        mock_response.encoding_format = "float"
        mock_response.dimensions = 1024
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        
        mock_client_instance.embed.return_value = mock_response
        
        embedder = MixedbreadDocumentEmbedder(api_key=Secret.from_token("test-key"))
        
        documents = [Document(content="Test document")]
        result = embedder.run(documents, prompt="Custom prompt:")
        
        mock_client_instance.embed.assert_called_once_with(
            model="mixedbread-ai/mxbai-embed-large-v1",
            input=["Test document"],
            normalized=True,
            encoding_format="float",
            dimensions=None,
            prompt="Custom prompt:",
        )

    @pytest.mark.asyncio
    @patch('mixedbread_ai_haystack.embedders.document_embedder.AsyncMixedbread')
    async def test_run_async_success(self, mock_async_client):
        """Test successful async document embedding."""
        # Mock the async client and response
        mock_client_instance = AsyncMock()
        mock_async_client.return_value = mock_client_instance
        
        mock_response = Mock()
        mock_response.model = "test-model"
        mock_response.usage.model_dump.return_value = {"prompt_tokens": 10, "total_tokens": 10}
        mock_response.normalized = True
        mock_response.encoding_format = "float"
        mock_response.dimensions = 1024
        mock_response.data = [
            Mock(embedding=[0.7, 0.8, 0.9]),
            Mock(embedding=[0.1, 0.2, 0.3])
        ]
        
        mock_client_instance.embed.return_value = mock_response
        
        embedder = MixedbreadDocumentEmbedder(api_key=Secret.from_token("test-key"))
        
        documents = [
            Document(content="Async document 1"),
            Document(content="Async document 2")
        ]
        
        result = await embedder.run_async(documents)
        
        # Check that embeddings are attached to documents
        assert len(result["documents"]) == 2
        assert result["documents"][0].embedding == [0.7, 0.8, 0.9]
        assert result["documents"][1].embedding == [0.1, 0.2, 0.3]
        assert result["meta"]["model"] == "test-model"
        
        # Verify async API call
        mock_client_instance.embed.assert_called_once_with(
            model="mixedbread-ai/mxbai-embed-large-v1",
            input=["Async document 1", "Async document 2"],
            normalized=True,
            encoding_format="float",
            dimensions=None,
            prompt=None,
        )

    @pytest.mark.asyncio
    @patch('mixedbread_ai_haystack.embedders.document_embedder.AsyncMixedbread')
    async def test_run_async_empty_documents(self, mock_async_client):
        """Test async embedding with empty document list."""
        mock_client_instance = AsyncMock()
        mock_async_client.return_value = mock_client_instance
        
        embedder = MixedbreadDocumentEmbedder(api_key=Secret.from_token("test-key"))
        
        result = await embedder.run_async([])
        
        assert result["documents"] == []
        assert result["meta"]["dimensions"] == 0
        
        # Should not call the API for empty documents
        mock_client_instance.embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_async_invalid_input(self):
        """Test async embedding with invalid input type."""
        embedder = MixedbreadDocumentEmbedder(api_key=Secret.from_token("test-key"))
        
        with pytest.raises(TypeError, match="Input must be a list of Haystack Documents"):
            await embedder.run_async(["not", "documents"])

    def test_create_metadata(self):
        """Test metadata creation from response."""
        embedder = MixedbreadDocumentEmbedder(api_key=Secret.from_token("test-key"))
        
        mock_response = Mock()
        mock_response.model = "test-model"
        mock_response.usage.model_dump.return_value = {"prompt_tokens": 20, "total_tokens": 20}
        mock_response.normalized = False
        mock_response.encoding_format = "float16"
        mock_response.dimensions = 512
        
        meta = embedder._create_metadata(mock_response)
        
        assert meta["model"] == "test-model"
        assert meta["usage"]["prompt_tokens"] == 20
        assert meta["normalized"] == False
        assert meta["encoding_format"] == "float16"
        assert meta["dimensions"] == 512

    @patch('mixedbread_ai_haystack.embedders.document_embedder.Mixedbread')
    def test_document_content_preservation(self, mock_client):
        """Test that original document content is preserved."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_response = Mock()
        mock_response.model = "test-model"
        mock_response.usage.model_dump.return_value = {"prompt_tokens": 5, "total_tokens": 5}
        mock_response.normalized = True
        mock_response.encoding_format = "float"
        mock_response.dimensions = 1024
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        
        mock_client_instance.embed.return_value = mock_response
        
        embedder = MixedbreadDocumentEmbedder(api_key=Secret.from_token("test-key"))
        
        original_doc = Document(content="Original content", meta={"key": "value"})
        documents = [original_doc]
        
        result = embedder.run(documents)
        
        # Check that original content and metadata are preserved
        embedded_doc = result["documents"][0]
        assert embedded_doc.content == "Original content"
        assert embedded_doc.meta == {"key": "value"}
        assert embedded_doc.embedding == [0.1, 0.2, 0.3]
        
        # Ensure it's the same document object (modified in place)
        assert embedded_doc is original_doc