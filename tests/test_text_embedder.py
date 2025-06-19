"""
Tests for MixedbreadTextEmbedder.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock

from haystack import Document
from haystack.utils import Secret

from mixedbread_ai_haystack import MixedbreadTextEmbedder


class TestMixedbreadTextEmbedder:
    
    def test_init(self):
        """Test embedder initialization."""
        embedder = MixedbreadTextEmbedder(
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
                MixedbreadTextEmbedder()

    def test_to_dict(self):
        """Test serialization to dictionary."""
        with patch.dict("os.environ", {"MXBAI_API_KEY": "test-key"}):
            embedder = MixedbreadTextEmbedder(model="test-model")
            
            data = embedder.to_dict()
            
            assert data["type"] == "mixedbread_ai_haystack.embedders.text_embedder.MixedbreadTextEmbedder"
            assert data["init_parameters"]["model"] == "test-model"
            assert "api_key" in data["init_parameters"]

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "type": "mixedbread_ai_haystack.embedders.text_embedder.MixedbreadTextEmbedder",
            "init_parameters": {
                "api_key": {"type": "env_var", "env_vars": ["MXBAI_API_KEY"], "strict": True},
                "model": "test-model",
                "normalized": False
            }
        }
        
        with patch.dict("os.environ", {"MXBAI_API_KEY": "test-key"}):
            embedder = MixedbreadTextEmbedder.from_dict(data)
            assert embedder.model == "test-model"
            assert embedder.normalized == False

    @patch('mixedbread_ai_haystack.embedders.text_embedder.Mixedbread')
    def test_run_success(self, mock_client):
        """Test successful text embedding."""
        # Mock the client and response
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
        
        embedder = MixedbreadTextEmbedder(api_key=Secret.from_token("test-key"))
        
        result = embedder.run("test text")
        
        assert result["embedding"] == [0.1, 0.2, 0.3]
        assert result["meta"]["model"] == "test-model"
        assert result["meta"]["normalized"] == True
        assert result["meta"]["dimensions"] == 1024
        
        # Verify API call
        mock_client_instance.embed.assert_called_once_with(
            model="mixedbread-ai/mxbai-embed-large-v1",
            input=["test text"],
            normalized=True,
            encoding_format="float",
            dimensions=None,
            prompt=None,
        )

    @patch('mixedbread_ai_haystack.embedders.text_embedder.Mixedbread')
    def test_run_empty_text(self, mock_client):
        """Test embedding empty text."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        embedder = MixedbreadTextEmbedder(api_key=Secret.from_token("test-key"))
        
        result = embedder.run("")
        
        assert result["embedding"] == []
        assert result["meta"]["model"] == "mixedbread-ai/mxbai-embed-large-v1"
        assert result["meta"]["dimensions"] == 0
        
        # Should not call the API for empty text
        mock_client_instance.embed.assert_not_called()

    @patch('mixedbread_ai_haystack.embedders.text_embedder.Mixedbread')
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
        
        embedder = MixedbreadTextEmbedder(api_key=Secret.from_token("test-key"))
        
        result = embedder.run("test text", prompt="Custom prompt:")
        
        mock_client_instance.embed.assert_called_once_with(
            model="mixedbread-ai/mxbai-embed-large-v1",
            input=["test text"],
            normalized=True,
            encoding_format="float",
            dimensions=None,
            prompt="Custom prompt:",
        )

    @pytest.mark.asyncio
    @patch('mixedbread_ai_haystack.embedders.text_embedder.AsyncMixedbread')
    async def test_run_async_success(self, mock_async_client):
        """Test successful async text embedding."""
        # Mock the async client and response
        mock_client_instance = AsyncMock()
        mock_async_client.return_value = mock_client_instance
        
        mock_response = Mock()
        mock_response.model = "test-model"
        mock_response.usage.model_dump.return_value = {"prompt_tokens": 5, "total_tokens": 5}
        mock_response.normalized = True
        mock_response.encoding_format = "float"
        mock_response.dimensions = 1024
        mock_response.data = [Mock(embedding=[0.4, 0.5, 0.6])]
        
        mock_client_instance.embed.return_value = mock_response
        
        embedder = MixedbreadTextEmbedder(api_key=Secret.from_token("test-key"))
        
        result = await embedder.run_async("async test text")
        
        assert result["embedding"] == [0.4, 0.5, 0.6]
        assert result["meta"]["model"] == "test-model"
        
        # Verify async API call
        mock_client_instance.embed.assert_called_once_with(
            model="mixedbread-ai/mxbai-embed-large-v1",
            input=["async test text"],
            normalized=True,
            encoding_format="float",
            dimensions=None,
            prompt=None,
        )

    @pytest.mark.asyncio
    @patch('mixedbread_ai_haystack.embedders.text_embedder.AsyncMixedbread')
    async def test_run_async_empty_text(self, mock_async_client):
        """Test async embedding with empty text."""
        mock_client_instance = AsyncMock()
        mock_async_client.return_value = mock_client_instance
        
        embedder = MixedbreadTextEmbedder(api_key=Secret.from_token("test-key"))
        
        result = await embedder.run_async("   ")  # Whitespace only
        
        assert result["embedding"] == []
        assert result["meta"]["dimensions"] == 0
        
        # Should not call the API for empty text
        mock_client_instance.embed.assert_not_called()

    def test_create_metadata(self):
        """Test metadata creation from response."""
        embedder = MixedbreadTextEmbedder(api_key=Secret.from_token("test-key"))
        
        mock_response = Mock()
        mock_response.model = "test-model"
        mock_response.usage.model_dump.return_value = {"prompt_tokens": 10, "total_tokens": 10}
        mock_response.normalized = False
        mock_response.encoding_format = "float16"
        mock_response.dimensions = 512
        
        meta = embedder._create_metadata(mock_response)
        
        assert meta["model"] == "test-model"
        assert meta["usage"]["prompt_tokens"] == 10
        assert meta["normalized"] == False
        assert meta["encoding_format"] == "float16"
        assert meta["dimensions"] == 512

    @patch('mixedbread_ai_haystack.embedders.text_embedder.Mixedbread')
    def test_custom_encoding_format(self, mock_client):
        """Test embedder with custom encoding format."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_response = Mock()
        mock_response.model = "test-model"
        mock_response.usage.model_dump.return_value = {"prompt_tokens": 5, "total_tokens": 5}
        mock_response.normalized = False
        mock_response.encoding_format = "float16"
        mock_response.dimensions = 256
        mock_response.data = [Mock(embedding=[0.1, 0.2])]
        
        mock_client_instance.embed.return_value = mock_response
        
        embedder = MixedbreadTextEmbedder(
            api_key=Secret.from_token("test-key"),
            encoding_format="float16",
            dimensions=256,
            normalized=False
        )
        
        result = embedder.run("test")
        
        mock_client_instance.embed.assert_called_once_with(
            model="mixedbread-ai/mxbai-embed-large-v1",
            input=["test"],
            normalized=False,
            encoding_format="float16",
            dimensions=256,
            prompt=None,
        )