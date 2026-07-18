"""M4 因子库CRUD单元测试 (临时目录, 不碰真实artifacts)."""
import numpy as np
import pandas as pd
import pytest

from factor_miner.library.store import FactorLibrary


class _FakeCfg:
    def __init__(self, root):
        self._root = root

    @property
    def artifacts_dir(self):
        return self._root


@pytest.fixture
def lib(tmp_path):
    return FactorLibrary(_FakeCfg(tmp_path))


def _mats():
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    f = pd.DataFrame(np.random.default_rng(0).normal(size=(10, 5)),
                     index=idx, columns=[str(i) for i in range(5)])
    ic = {"10": pd.Series(np.linspace(0.01, 0.05, 10), index=idx)}
    return f, ic


METRICS = {"h10_train": {"icir": 0.5, "ic_mean": 0.03}, "coverage": 0.9, "n_nodes": 5}


def test_add_get_autoname(lib):
    f, ic = _mats()
    fid = lib.add("cs_rank(close)", "k1", "GP", METRICS, f, ic)
    d = lib.get(fid)
    assert d["name"].startswith("GP_") and d["metrics"]["h10_train"]["icir"] == 0.5
    assert lib.exists("k1") and not lib.exists("k2")
    pd.testing.assert_frame_equal(lib.load_values(fid).astype(float),
                                  f.astype("float32").astype(float),
                                  check_freq=False)


def test_list_sorted_by_icir(lib):
    f, ic = _mats()
    lib.add("a(close)", "ka", "GP", {"h10_train": {"icir": 0.3, "ic_mean": 0.02}}, f, ic)
    lib.add("b(close)", "kb", "RL", {"h10_train": {"icir": -0.8, "ic_mean": -0.04}}, f, ic)
    df = lib.list()
    assert df.iloc[0]["expr_key"] == "kb"       # |ICIR|降序
    assert set(df["engine"]) == {"GP", "RL"}


def test_update_and_soft_hard_delete(lib):
    f, ic = _mats()
    fid = lib.add("c(close)", "kc", "GP", METRICS, f, ic)
    lib.update(fid, name="我的反转因子", tags="反转,量价", notes="test", status="candidate")
    d = lib.get(fid)
    assert d["name"] == "我的反转因子" and d["status"] == "candidate"
    lib.delete(fid)                              # 软删
    assert lib.get(fid)["status"] == "archived"
    lib.delete(fid, hard=True)                   # 硬删
    assert lib.get(fid) is None
    assert not (lib.values_dir / f"{fid}.parquet").exists()


def test_duplicate_expr_key_rejected(lib):
    f, ic = _mats()
    lib.add("d(close)", "kd", "GP", METRICS, f, ic)
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        lib.add("d2(close)", "kd", "RL", METRICS, f, ic)
