"""
Base Lookup Interface
Defines the interface for all lookup strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseLookup(ABC):
    """
    Abstract base class for lookup strategies.
    All lookup implementations must inherit from this.
    """
    
    @abstractmethod
    def lookup(self, key: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Look up data by key.
        
        Args:
            key: The lookup key
            **kwargs: Additional parameters for lookup
            
        Returns:
            Dictionary with lookup results or None if not found
        """
        pass
    
    @abstractmethod
    def lookup_batch(self, keys: list, **kwargs) -> list:
        """
        Look up multiple items at once.
        
        Args:
            keys: List of lookup keys
            **kwargs: Additional parameters for lookup
            
        Returns:
            List of lookup results
        """
        pass