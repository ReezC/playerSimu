# -*- coding: utf-8 -*-
"""临时：跑一条 minimap 用例（RV 也用它 ✓）。用法：python tools/_run_one.py <用例名>"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import selftest_minimap as M                        # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "t_dot_basemap_layer_first"
names = sys.argv[1:] or [name]
for _n in names:
    try:
        getattr(M, _n)()
        print("RESULT OK %s" % _n)
    except AssertionError as e:
        print("RESULT NG %s :: %s" % (_n, str(e)[:260]))
    except Exception as e:                                     # noqa: BLE001
        import traceback
        traceback.print_exc()
        print("RESULT EX %s :: %s" % (_n, type(e).__name__))
