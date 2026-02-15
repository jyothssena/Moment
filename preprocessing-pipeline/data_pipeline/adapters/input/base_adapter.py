"""
Base Input Adapter Interface
Defines the interface for all input adapters.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any


class BaseInputAdapter(ABC):
    """
    Abstract base class for input adapters.
    All input implementations must inherit from this.
    """
    
    @abstractmethod
    def read(self) -> Iterator[Dict[str, Any]]:
        """
        Read data from source and yield one record at a time.
        
        Yields:
            Dictionary with record data
        """
        pass
    
    @abstractmethod
    def read_all(self) -> list:
        """
        Read all data from source at once.
        
        Returns:
            List of dictionaries with all records
        """
        pass