from hydra import compose, initialize
from omegaconf import OmegaConf


class ConfigManager:
    """
    ConfigManager allows loading and reloading Hydra configs on the fly.
    Usage:
        config = ConfigManager()
        cfg = config.load(verbose=True)
        # ... later ...
        cfg = config.reload(verbose=True)
    """
    def __init__(self, config_name="config", config_path="./configs", overrides=None, version_base="1.3.2"):
        self.config_name = config_name
        self.config_path = config_path
        self.overrides = overrides or ["optimizer/search_algo=mcts_feedback"]
        self.version_base = version_base
        self._cfg = None

    def load(self, overrides=None, verbose=False):
        """Load configuration using Hydra"""
        with initialize(version_base=self.version_base, config_path=self.config_path):
            self._cfg = compose(config_name=self.config_name, overrides=overrides or self.overrides)
            if verbose:
                print(OmegaConf.to_yaml(self._cfg))
            return self._cfg

    def reload(self, overrides=None, verbose=False):
        """Reload configuration using Hydra"""
        return self.load(overrides=overrides, verbose=verbose)

    @property
    def cfg(self):
        """Get the current config object (load if not loaded)"""
        if self._cfg is None:
            return self.load()
        return self._cfg
