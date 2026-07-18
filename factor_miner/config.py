"""全局配置加载：config/settings.toml (TOML, 零第三方依赖)。"""
from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Config:
    """settings.toml 的轻量包装，支持 cfg['data']['start_date'] 与路径解析。"""

    def __init__(self, path: str | Path | None = None):
        p = Path(path) if path else PROJECT_ROOT / "config" / "settings.toml"
        with open(p, "rb") as f:
            self._raw = tomllib.load(f)
        self.path = p

    def __getitem__(self, key: str):
        return self._raw[key]

    def get(self, section: str, key: str, default=None):
        return self._raw.get(section, {}).get(key, default)

    def _resolve(self, raw: str) -> Path:
        p = Path(raw)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def cache_dir(self) -> Path:
        return self._resolve(self._raw["data"]["cache_dir"])

    @property
    def artifacts_dir(self) -> Path:
        return self._resolve(self._raw["paths"]["artifacts_dir"])

    @property
    def raw_dir(self) -> Path:
        return self.cache_dir / "raw"

    @property
    def features_dir(self) -> Path:
        return self.cache_dir / "features"


@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config()
