from abc import ABC, abstractmethod

class Command(ABC):
    @abstractmethod
    def execute(self, **kwargs):
        """Method to execute the command"""
        pass