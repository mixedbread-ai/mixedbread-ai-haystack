from enum import Enum


class MixedbreadEmbeddingType(Enum):
    """
    Supported encoding formats for Mixedbread Embeddings API.
    """

    FLOAT = "float"
    FLOAT16 = "float16"
    BASE64 = "base64"
    BINARY = "binary"
    UBINARY = "ubinary"
    INT8 = "int8"
    UINT8 = "uint8"

    def __str__(self) -> str:
        """Return the string value of the embedding type."""
        return self.value

    @staticmethod
    def from_str(s: str) -> "MixedbreadEmbeddingType":
        """
        Convert a string to a MixedbreadEmbeddingType enum.
        
        Args:
            s: String representation of the embedding type.
            
        Returns:
            Corresponding MixedbreadEmbeddingType enum value.
            
        Raises:
            ValueError: If the string is not a valid embedding type.
        """
        try:
            return MixedbreadEmbeddingType(s.lower())
        except ValueError as e:
            raise ValueError(
                f"Unknown Mixedbread embedding type '{s}'. Supported types are: {[e.value for e in MixedbreadEmbeddingType]}"
            ) from e
