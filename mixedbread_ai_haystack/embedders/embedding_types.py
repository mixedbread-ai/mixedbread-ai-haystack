from enum import Enum


class MixedbreadEmbeddingType(Enum):
    """
    Supported encoding formats for Mixedbread AI embeddings.
    """

    FLOAT = "float"
    FLOAT16 = "float16"
    BASE64 = "base64"
    BINARY = "binary"
    UBINARY = "ubinary"
    INT8 = "int8"
    UINT8 = "uint8"

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def from_str(s: str) -> "MixedbreadEmbeddingType":
        try:
            return MixedbreadEmbeddingType(s.lower())
        except ValueError as e:
            raise ValueError(
                f"Unknown Mixedbread embedding type '{s}'. Supported types are: {[e.value for e in MixedbreadEmbeddingType]}"
            ) from e
