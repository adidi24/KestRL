from typing import Dict, Optional
import logging

from .base import EnvironmentBuilder
from .builders.gym_builder import GymEnvironmentFactory
from .builders.metaworld_builder import MetaWorldEnvironmentFactory


logger = logging.getLogger(__name__)

# Global environment builder instance
_global_env_builder: Optional[EnvironmentBuilder] = None


def get_env_builder() -> EnvironmentBuilder:
    """Get or create the global environment builder."""
    global _global_env_builder
    
    if _global_env_builder is None:
        _global_env_builder = EnvironmentBuilder()
        
        # Register default factories
        _global_env_builder.register_factory("gym", GymEnvironmentFactory())
        
        # Register optional factories
        try:
            _global_env_builder.register_factory("metaworld", MetaWorldEnvironmentFactory())
            logger.info("MetaWorld environment factory registered")
        except ImportError:
            logger.debug("MetaWorld not available")

        # Register healthcare factory
        try:
            from .builders.healthcare_builder import HealthcareEnvironmentFactory
            _global_env_builder.register_factory("healthcare", HealthcareEnvironmentFactory())
            logger.info("Healthcare environment factory registered")
        except ImportError as e:
            logger.debug(f"Healthcare environments not available: {e}")
        
        # Add more factories as needed
    
    return _global_env_builder


def register_custom_factory(name: str, factory):
    """Register a custom environment factory."""
    builder = get_env_builder()
    builder.register_factory(name, factory)