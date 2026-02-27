"""
Generic registry utilities for the synthetic module.

Provides a simple registry pattern for extensibility.
"""

from typing import Any, TypeVar

T = TypeVar("T")


class Registry(dict[str, T]):
    """
    A simple registry that maps string names to values.

    Subclass of dict with a register() method for convenient registration.
    """

    def register(self, name: str, value: T) -> None:
        """
        Register a value with a name.

        Args:
            name: Unique name for this entry
            value: Value to register

        Raises:
            ValueError: If name is already registered
        """
        if name in self:
            raise ValueError(f"'{name}' is already registered.")
        self[name] = value

    def get_or_raise(self, name: str) -> T:
        """
        Get a registered value by name, raising if not found.

        Args:
            name: Name to look up

        Returns:
            The registered value

        Raises:
            ValueError: If name is not registered
        """
        if name not in self:
            raise ValueError(f"Unknown name: '{name}'. Available: {list(self.keys())}")
        return self[name]


# Registry for firing probability config classes
FIRING_PROB_REGISTRY: Registry[type[Any]] = Registry()

# Registry for magnitude config classes
MAGNITUDE_REGISTRY: Registry[type[Any]] = Registry()
