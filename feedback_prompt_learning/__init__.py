__version__ = "0.1.0"
from feedback_prompt_learning.config import ConfigManager
from feedback_prompt_learning.search_algo import *  # noqa: F401, F403

config = ConfigManager()
config.load()

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "cfg",
    "load_config",
]
