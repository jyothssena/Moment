"""
Base Output Adapter Interface
Defines the interface for all output adapters.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseOutputAdapter(ABC):
    """
    Abstract base class for output adapters.
    All output implementations must inherit from this.
    """
    
    @abstractmethod
    def write(self, data: Dict[str, Any]) -> None:
        """
        Write a single record.
        
        Args:
            data: Dictionary with record data
        """
        pass
    
    @abstractmethod
    def write_batch(self, data: List[Dict[str, Any]]) -> None:
        """
        Write multiple records at once.
        
        Args:
            data: List of dictionaries with record data
        """
        pass
    
    @abstractmethod
    def finalize(self) -> None:
        """
        Finalize writing (close files, commit transactions, etc.).
        """
        pass