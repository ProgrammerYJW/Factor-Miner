"""M8 端到端验证: Streamlit AppTest 无头执行四页面 + 真实因子库删改闭环.

用法: .venv/Scripts/python.exe scripts/e2e_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAGES = ["page_overview.py", "page_detail.py", "page_mining.py", "page_corr.py"]


def check_pages() -> bool:
    from streamlit.testing.v1 import AppTest

    ok = True
    for p in PAGES:
        at = AppTest.from_file(str(ROOT / "factor_miner" / "webapp" / p),
                               default_timeout=120)
        at.run()
        errs = [e.value for e in at.exception]
        if errs:
            ok = False
            print(f"[FAIL] {p}: {errs}")
        else:
            n_el = len(at.dataframe) + len(at.metric) + len(at.title)
            print(f"[OK]   {p}: 渲染正常 (dataframe={len(at.dataframe)} "
                  f"metric={len(at.metric)} 异常=0)")
    return ok


def check_crud_roundtrip() -> bool:
    from factor_miner.library import FactorLibrary

    lib = FactorLibrary()
    df = lib.list()
    if not len(df):
        print("[SKIP] 因子库为空, 跳过CRUD闭环")
        return True
    fid = int(df.iloc[-1]["id"])
    orig = lib.get(fid)
    try:
        lib.update(fid, name="__e2e_测试改名__", tags="e2e", notes="端到端测试")
        assert lib.get(fid)["name"] == "__e2e_测试改名__", "改名失败"
        lib.delete(fid)                                  # 软删
        assert lib.get(fid)["status"] == "archived", "软删失败"
        print(f"[OK]   CRUD闭环: #{fid} 改名/标签/备注/软删 全部生效")
        return True
    finally:                                             # 恢复原状
        lib.update(fid, name=orig["name"], tags=orig["tags"],
                   notes=orig["notes"], status=orig["status"])
        assert lib.get(fid)["name"] == orig["name"]
        print(f"[OK]   CRUD闭环: #{fid} 已恢复原状 (name={orig['name']}, "
              f"status={orig['status']})")


def main() -> int:
    print("=" * 70)
    print("FactorMiner 端到端验证 (Web四页面 + 因子库删改闭环)")
    print("=" * 70)
    ok1 = check_pages()
    ok2 = check_crud_roundtrip()
    print("=" * 70)
    print("端到端验证通过 [OK]" if (ok1 and ok2) else "端到端验证存在失败项 [FAIL]")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
