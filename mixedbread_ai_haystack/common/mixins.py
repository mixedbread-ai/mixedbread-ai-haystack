"""Mixins for common component functionality."""

from typing import Dict, Any, Type
from haystack import default_to_dict, default_from_dict
from haystack.utils import deserialize_secrets_inplace

from .client import MixedbreadClient


class SerializationMixin:
    """Mixin to provide standard serialization/deserialization for components."""
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize component to dictionary.
        
        Returns:
            Dictionary representation of the component.
        """
        if not isinstance(self, MixedbreadClient):
            raise TypeError("SerializationMixin can only be used with MixedbreadClient subclasses")
            
        client_params = MixedbreadClient.to_dict(self)["init_parameters"]
        
        # Get all init parameters except those from MixedbreadClient
        component_params = {}
        for attr in dir(self):
            if not attr.startswith('_') and hasattr(self, attr):
                value = getattr(self, attr)
                # Skip methods and client-specific attributes
                if (not callable(value) and 
                    attr not in ['client', 'api_key', 'base_url', 'timeout', 'max_retries']):
                    component_params[attr] = value
        
        return default_to_dict(self, **client_params, **component_params)
    
    @classmethod 
    def from_dict(cls: Type, data: Dict[str, Any]):
        """
        Deserialize component from dictionary.
        
        Args:
            data: Dictionary containing component data.
            
        Returns:
            Instantiated component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)