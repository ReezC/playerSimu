"""宠物图库 + **组合外观模板**自检（用户 2026-10-04 ✓ 方案 C）。

跑法：`python -m tools.selftest_pets`

⚠ 图库本身现在**还没导出**（要 `WzProbe` 加 `dump-pets` / `dump-petequips` 两个通道 ✓）——
  所以这里用**假数据**：摸清规则靠假数据就够 ✓，真数据来了直接套 ✓。
  真数据那一路的"缺什么就说清什么"由 `pet_library_state()` 负责 ✓（本文件第 ① 组钉它 ✓）。
"""
import json
import pathlib
import shutil
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import cv2                                                          # noqa: E402

from core import petlib, wzexport                                   # noqa: E402

FAILED = []


def check(ok, msg):
    if not ok:
        raise AssertionError(msg)


def _png(path, bgr, size=(2, 2), alpha=255):
    """造一张测试图（BGR 三通道写盘 ⇒ cv2 读回来会补成 BGRA ✓ 正好走我们那条路 ✓）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((size[1], size[0], 3), np.uint8)
    img[:, :] = bgr
    check(cv2.imwrite(str(path), img), "造图失败：%s" % path)
    return path


#: `petlib` 那几个"路径函数"的**原始值**（第一次造 `_Lib` 时记下 ✓ 见 `restore` ✓）。
_ORIG = None


class _Lib:
    """造一套**假的**宠物/装备图库（用完把 monkeypatch 还原 ✓）。"""

    def __init__(self, root, pets=("5000028",), equips=(("01802000", ("5000028",)),),
                 with_index=True, with_equip_index=True):
        self.root = pathlib.Path(root)
        self.pet = self.root / "pet"
        self.eq = self.root / "pet_equip"
        self.combo = self.root / "combo"
        self.pet_idx = self.root / "pets.json"
        self.eq_idx = self.root / "pet_equips.json"
        for pid in pets:
            _png(self.pet / pid / "stand0_0.png", (0, 0, 255))      # 红（BGR）✓ 2×2
            _png(self.pet / pid / "stand0_1.png", (0, 0, 200))
            _png(self.pet / pid / "icon.png", (0, 0, 255), size=(1, 1))
            (self.pet / pid / "meta.tsv").write_text(
                "#file\toriginX\toriginY\tltX\tltY\trbX\trbY\tdelay\n"
                "stand0_0.png\t1\t2\t-1\t-2\t1\t0\t180\n"
                "stand0_1.png\t1\t2\t-1\t-2\t1\t0\t180\n", encoding="utf-8")
        for eid, can in equips:
            # ⭐ 装备的帧**按宠物分开**（真实布局 = `PetEquip/<装备>/<宠物id>/stand0/<n>` ✓
            #   见 WzProbe/PetDump.cs ✓）⇒ 假数据也得长这样 ✓，否则测不到"能戴/戴不了" ✓
            _png(self.eq / eid / "icon.png", (255, 0, 0), size=(1, 1))
            #   ⭐ `meta.tsv` 在**装备那一层** ✓，行名带 `<宠物id>/` 前缀 ✓（与导出同构 ✓
            #     见 WzProbe/PetDump.cs ✓）—— 不照这个写，就测不到"origin 会不会丢" ✓
            _rows = "\n".join("%s/stand0_0.png\t0\t0\t0\t0\t0\t0\t0" % p for p in (can or ()))
            (self.eq / eid / "meta.tsv").write_text(
                "#file\toriginX\toriginY\tltX\tltY\trbX\trbY\tdelay\n" + _rows + "\n",
                encoding="utf-8")
            for pid in (can or ()):
                _png(self.eq / eid / pid / "stand0_0.png", (255, 0, 0), size=(1, 1))
        if with_index:
            self.pet_idx.write_text(json.dumps(
                [{"id": p, "name": "进化龙"} for p in pets], ensure_ascii=False), encoding="utf-8")
        if with_equip_index:
            self.eq_idx.write_text(json.dumps(
                [{"id": e, "name": "红丝带", "pets": list(c)} for e, c in equips],
                ensure_ascii=False), encoding="utf-8")
        # ⚠⚠ **只记第一次见到的那份**（模块级 ✓）—— 一个用例里连着造好几个 `_Lib` 时，
        #   各自记"创建时的值"会把**上一个的补丁**当成原始值 ✗ ⇒ `restore()` 还原不回去 ✓
        #   （实测后果：后面的"真图库"那条用例看到的是**已经删掉的临时目录** ⇒ 被当成
        #    "图库还没导出" 而静默跳过 ✗ —— 用例自己的 bug ✓ 但很隐蔽 ✓）。
        global _ORIG
        if _ORIG is None:
            _ORIG = (petlib.pet_dir, petlib.pet_equip_dir, petlib.pet_index_path,
                     petlib.pet_equip_index_path, petlib.combo_root,
                     petlib.pet_name_fixes_path)
        self._old = _ORIG
        petlib.pet_dir = lambda: self.pet
        petlib.pet_equip_dir = lambda: self.eq
        petlib.pet_index_path = lambda: self.pet_idx
        petlib.pet_equip_index_path = lambda: self.eq_idx
        petlib.combo_root = lambda: self.combo
        # ⭐ 手改表也要指到**临时目录**（不然真表会漏进假图库的用例里 ✓ 见 `pet_name_fixes_path` ✓）
        self.fixes = self.root / "pets_name_fixes.json"
        petlib.pet_name_fixes_path = lambda: self.fixes
        petlib.clear_pet_caches()

    def restore(self):
        (petlib.pet_dir, petlib.pet_equip_dir, petlib.pet_index_path,
         petlib.pet_equip_index_path, petlib.combo_root,
         petlib.pet_name_fixes_path) = self._old
        petlib.clear_pet_caches()


def _tmpdir():
    return tempfile.mkdtemp(prefix="pets_")


# ══════════════════════════════════════════════════════════════
def t_library_state():
    """① 「图库还没导出」必须**如实说缺哪一步**（照掉落物那套 ✓ 用户 2026-10-04 ✓）。

    钉三件：① 全缺 ⇒ 说"要 WzProbe 加两个通道"（点名 `dump-pets` / `dump-petequips` ✓
    别只说"没有候选项"✗）；② 只有宠物 ⇒ 说清"装备清单是空的"；③ 齐全 ⇒ 报条数 ✓。
    """
    tmp = _tmpdir()
    try:
        # ① 全缺
        lib = _Lib(tmp, pets=(), equips=(), with_index=False, with_equip_index=False)
        ok, why = petlib.pet_library_state()
        check(ok is False and "dump-pets" in why and "dump-petequips" in why,
              "图库全缺时没说清缺什么 / 该给 WzProbe 加哪两个通道（用户要求遍历补全那个思路 ✗）："
              "%r / %r" % (ok, why[:200]))
        shutil.rmtree(tmp, ignore_errors=True)

        # ② 只有宠物（没有装备清单）
        tmp = _tmpdir()
        lib = _Lib(tmp, with_equip_index=False)
        ok, why = petlib.pet_library_state()
        check(ok is False and "装备" in why,
              "只有宠物、没有装备清单时该说清（右栏会空着 ✗）：%r / %r" % (ok, why))
        check(len(petlib.load_pet_index()) == 1
              and petlib.load_pet_index()[0]["name"] == "进化龙",
              "宠物清单读不出来：%r" % (petlib.load_pet_index(),))
        shutil.rmtree(tmp, ignore_errors=True)

        # ③ 齐全
        tmp = _tmpdir()
        lib = _Lib(tmp)
        ok, why = petlib.pet_library_state()
        check(ok is True and "宠物 1 项" in why and "装备 1 项" in why,
              "图库齐了却没报条数：%r / %r" % (ok, why))
        rows = petlib.load_pet_equip_index()
        check(rows and rows[0]["pets"] == ["5000028"],
              "装备清单里的「能戴它的宠物」没读出来（右栏就靠它 ✗）：%r" % (rows,))
        check(petlib.sprite_icon(petlib.pet_dir(), "5000028") is not None,
              "列表图标（icon.png）没找到 ✗")
        check(petlib.sprite_icon(petlib.pet_dir(), "没有这个") is None,
              "没有图标的 id 该给 None（界面画空图标 ✓ 不许回退成别的图 ✗）")
    finally:
        lib.restore()
        shutil.rmtree(tmp, ignore_errors=True)


def t_frames_and_meta():
    """② 帧 + origin：只认 `<state>_<n>.png`（`icon.png` 不算帧 ✓）、按帧号排序 ✓、
    `meta.tsv` 缺/坏 ⇒ 退回"以中心为原点"（**同 `tools/cutpaste.py` 的老口径** ✓ 不许抛 ✗）。
    """
    tmp = _tmpdir()
    lib = _Lib(tmp)
    try:
        fs = petlib.sprite_frames(petlib.pet_dir(), "5000028")
        check([f.name for f, _ox, _oy in fs] == ["stand0_0.png", "stand0_1.png"],
              "帧列得不对（该只收 stand0_*、按帧号排序 ✓）：%r"
              % ([f.name for f, _ox, _oy in fs],))
        check(fs[0][1] == 1.0 and fs[0][2] == 2.0,
              "meta.tsv 的 origin 没读对：%r" % (fs[0][1:],))
        check(petlib.sprite_frames(petlib.pet_dir(), "5000028", "move0") == [],
              "没有那个动作时该给空表（别编 ✗）")
        # 坏 meta ⇒ 退回中心原点
        (petlib.pet_dir() / "5000028" / "meta.tsv").write_text("坏数据\n", encoding="utf-8")
        fs2 = petlib.sprite_frames(petlib.pet_dir(), "5000028")
        check(len(fs2) == 2 and fs2[0][1] is None,
              "meta 坏掉时该给 None（调用方退回中心 ✓）而不是抛 ✗：%r" % (fs2,))
    finally:
        lib.restore()
        shutil.rmtree(tmp, ignore_errors=True)


def t_compose():
    """③ ⭐⭐ **组合外观**（方案 C 的心脏 ✓）—— 照 MapleNecrocer `Client/Pet.cs` 的规则：
    `class PetEquip : Pet` ⇒ **同锚点、各按自己那帧的 origin 摆** ✓ 组合 = 按 origin 对齐叠 ✓。

    这条用**可判定的几何**钉死（假图 + 已知 origin ⇒ 画布尺寸/origin/像素位置都能算出来 ✓）：
      · 宠物 2×2、origin(1,2) ⇒ 在锚点坐标里占 x∈[-1,1] y∈[-2,0] ✓；
      · 装备 1×1、origin(0,0) ⇒ 占 x∈[0,1] y∈[0,1] ✓；
      · 并集 ⇒ **2×3**、组合 origin = **(1,2)** ✓，装备那一像素应落在画布 (列1, 行2) ✓。
    另钉：装备帧比宠物少 ⇒ **按 `i % 装备帧数` 取**（不许整只不标 ✗）；宠物没这动作 ⇒ `None` ✓；
      缓存（源没变不重做 ✓ 源变了要重做 ✓）；产物格式与怪物精灵库一致（`meta.tsv` ✓ + `meta.json` ✓）。
    """
    tmp = _tmpdir()
    lib = _Lib(tmp)
    try:
        out = petlib.compose_pet("5000028", ["01802000"])
        check(out is not None and out.is_dir(), "组合做不出来：%r" % (out,))
        check(out.name == petlib.combo_key("5000028", ["01802000"]) == "5000028__01802000",
              "组合目录名不对（要可复现 ✓）：%r" % (out.name,))
        img = cv2.imread(str(out / "stand0_0.png"), cv2.IMREAD_UNCHANGED)
        check(img is not None and img.shape[:2] == (3, 2),
              "组合画布尺寸不对（宠物 2×2 + 装备 1×1 按 origin 摆 ⇒ 该 2×3 ✓）：%r"
              % (None if img is None else img.shape,))
        b, _g, r = img[2, 1][:3]                 # 第 2 行第 1 列 = 装备那一格 ✓
        check(int(b) > 200 and int(r) < 60,
              "装备没叠在正确位置（该落在 (列1,行2) ✓）：BGR=%r" % (img[2, 1][:3],))
        b0, _g0, r0 = img[0, 0][:3]
        check(int(r0) > 200 and int(b0) < 60, "宠物本体那一格没了：BGR=%r" % (img[0, 0][:3],))
        meta = (out / "meta.tsv").read_text(encoding="utf-8")
        check("stand0_0.png\t1\t2\t" in meta and "stand0_1.png\t1\t2\t" in meta,
              "组合的 meta.tsv 格式/origin 不对（下游模板匹配就吃这个 ✗）：%r" % (meta,))
        mj = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        check(mj.get("pet") == "5000028" and mj.get("equips") == ["01802000"]
              and mj.get("frames") == 2,
              "meta.json 没记清「是宠物 + 哪些装备」（界面要回读它 ✗）：%r" % (mj,))
        # 装备只有 1 帧、宠物 2 帧 ⇒ 第 2 帧也得有**而且还得戴着那只装备**（按 i % 1 取 ✓）
        #   ⚠ 只断言"文件在"是不够的 ✗ —— 实现若"帧号对不上就不叠装备"，文件照样在、
        #     人却光着身子 ✗（反向验证里就是这么被蒙过去的 ✓）。
        check((out / "stand0_1.png").is_file(),
              "装备帧比宠物少时，宠物那几帧都得产出（按 i % 装备帧数 取 ✓ 不许少标 ✗）")
        img1 = cv2.imread(str(out / "stand0_1.png"), cv2.IMREAD_UNCHANGED)
        # ⚠ 越界也要给**一句人话**（反向验证时这里直接 IndexError ✗ 看不出是哪条规矩坏了 ✓）
        check(img1 is not None and img1.shape[:2] == (3, 2),
              "第 2 帧的画布尺寸也不对（装备该照样叠上去 ⇒ 还是 2×3 ✓）：%r"
              % (None if img1 is None else img1.shape,))
        b1, _g1, r1 = img1[2, 1][:3]
        check(int(b1) > 200 and int(r1) < 60,
              "第 2 帧上装备没了（缺帧该按 `i %% 装备帧数` 取模复用 ✓ 不许「对不上就不叠」✗）："
              "BGR=%r" % (img1[2, 1][:3],))

        # 缓存：源没变 ⇒ 不重做（产物 mtime 不变 ✓）
        t0 = (out / "stand0_0.png").stat().st_mtime_ns
        again = petlib.compose_pet("5000028", ["01802000"])
        check(again == out and (out / "stand0_0.png").stat().st_mtime_ns == t0,
              "同一组合重复调用重做了（白费功夫 ✗）")
        # 源变了 ⇒ 要重做（给装备补一帧 ⇒ 指纹变 ✓）
        _png(petlib.pet_equip_dir() / "01802000" / "5000028" / "stand0_1.png",
             (0, 255, 0), size=(1, 1))
        petlib.compose_pet("5000028", ["01802000"])
        check((out / "stand0_0.png").stat().st_mtime_ns != t0,
              "装备目录变了（加了帧）却没重做组合 ⇒ 模板过期 ✗")

        # 宠物没有这动作 ⇒ None（调用方如实说 ✓）
        check(petlib.compose_pet("5000028", ["01802000"], state="rest1") is None,
              "宠物没有那个动作时该给 None（别产出空模板 ✗）")
        # 组合也认「没戴装备」这一种（= 只有本体 ✓）
        solo = petlib.compose_pet("5000028")
        check(solo is not None and solo.name == "5000028",
              "不戴装备那种组合做不出来：%r" % (solo,))
        check("5000028__01802000" in petlib.combos_state(),
              "`combos_state()` 没列出已做好的组合：%r" % (petlib.combos_state(),))
        # 删掉重做
        check(petlib.drop_combo("5000028", ["01802000"]) is True
              and not (out / "meta.tsv").exists(),
              "`drop_combo` 没删掉产物 ⇒ 改了装备想强制重做时没抓手 ✗")
    finally:
        lib.restore()
        shutil.rmtree(tmp, ignore_errors=True)


def t_real_library_and_compose():
    """④ ⭐⭐ **用真导出的图库走一遍**（假数据只能验规则 ✓ 真数据才验"接没接对" ✓）。

    这一条就是被"真数据"抓出来的那种事：假数据里我把装备的 `meta.tsv` 放在
    `<装备>/<宠物id>/` 下 ✗，而**真导出**是放在 **`<装备>/` 那一层、行名带 `<宠物id>/` 前缀** ✓
    ⇒ 按假数据写的代码会把**所有装备帧的 origin 全丢掉** ✗（组合叠歪 ✓ 实测抓到 ✓）。
    ⚠ 图库**没导出**（`WzProbe dump-pets/dump-petequips` 还没跑 ✓）⇒ **明确跳过** ✓
      （不是"静默通过"✗ —— 打印一行说清跳过原因 ✓ 同 `t_drop_e2e` 的做法 ✓）。
    """
    ok, why = petlib.pet_library_state()
    if not ok:
        print("       （跳过：宠物图库还没导出 —— %s）" % why.splitlines()[0])
        return
    pets = petlib.load_pet_index()
    equips = petlib.load_pet_equip_index()
    check(len(pets) > 0 and all(p["id"] and p["name"] for p in pets[:5]),
          "真宠物清单不完整（有 id 没名字 ⇒ 界面没法搜 ✗）：%r" % (pets[:3],))
    check(any((e.get("pets") or []) for e in equips),
          "真装备清单里没有「能戴它的宠物」⇒ 右栏会是空的 ✗：%r" % (equips[:2],))

    eq = max(equips, key=lambda e: len(e.get("pets") or []))
    pid = eq["pets"][0]

    # ⭐⭐ **装备帧的 origin 必须在**（这条直指那个真 bug ✓）：装备的 `meta.tsv` 在 `<装备>/`
    #   那一层、行名带 `<宠物id>/` 前缀 ✓ ⇒ 少认一处，origin 就全 None ✗ ⇒ 组合**叠歪**
    #   （而"有差异"这种弱断言照样过 ✗ —— 反向验证实测：两条改坏都还是 PASS ✗）
    #   ⇒ 所以这里直接钉"取到的帧有没有 origin" ✓ 那才是要害 ✓。
    fs = petlib.sprite_frames(petlib.pet_equip_dir(), eq["id"], petlib.DEFAULT_STATE, pet_id=pid)
    check(fs, "真装备的帧一帧都没取到（能戴关系/目录层级对不上 ✗）：%r" % (eq["id"],))
    check(all(ox is not None and oy is not None for _f, ox, oy in fs),
          "装备帧的 **origin 丢了** ⇒ 组合会叠歪（`meta.tsv` 在上一层、行名带 `<宠物id>/` 前缀 ✗）：%r"
          % ([(f.name, ox, oy) for f, ox, oy in fs[:3]],))

    solo = petlib.compose_pet(pid)
    pair = petlib.compose_pet(pid, [eq["id"]])
    check(solo is not None and pair is not None,
          "真数据做不出组合（宠物 %s + 装备 %s ✗）：%r / %r" % (pid, eq["id"], solo, pair))
    a = cv2.imread(str(solo / "stand0_0.png"), cv2.IMREAD_UNCHANGED)
    b = cv2.imread(str(pair / "stand0_0.png"), cv2.IMREAD_UNCHANGED)
    check(a is not None and b is not None, "真组合的图读不出来 ✗")
    # ⭐ 装备**真的叠上去了**：组合图和"只宠物"必须有差异 ✓（尺寸也可能变大 ✓）
    h, w = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
    pa = np.zeros((h, w, 4), np.uint8)
    pb = np.zeros((h, w, 4), np.uint8)
    pa[:a.shape[0], :a.shape[1]] = a
    pb[:b.shape[0], :b.shape[1]] = b
    diff = int(np.count_nonzero(np.any(pa != pb, axis=2)))
    check(diff > 0 and (b.shape != a.shape or diff > 0),
          "宠物 %s + 装备 %s 的组合图和「只宠物」一模一样 ⇒ 装备没叠上去 ✗" % (pid, eq["id"]))
    check("stand0_0.png" in (pair / "meta.tsv").read_text(encoding="utf-8"),
          "真组合的 meta.tsv 不对（下游模板匹配就吃它 ✗）")


def t_pet_picker_flow():
    """⑤ 弹窗流程：**搜索筛选宠物 → 选这只宠物戴的装备 → 加入列表**（用户 2026-10-04 ✓ 原话 ✓）。

    钉五件：
      ① 左栏列出宠物（带图标 ✓）；右栏**只列这只宠物能戴的**（真源 = 清单里的 `pets` ✓）；
      ② ⭐ **选 1 件装备就只加 1 件** —— `IconGrid.ids()` 给的是**格子里全部** id ✗
         （拿它当"选中"会一加加一串 ✓ 本轮第一版就是这么写的 ✓ 冒烟一试就露馅 ✓）；
      ③ 加入的那一套 = `{"pet", "name", "equips"}` ✓（卡片直接存它 ✓）；
      ④ 同一套**不重复加** ✓；同一只宠物**可以配不同装备各加一套** ✓；
      ⑤ 「移出选中」按**那一套**删（同一只配了两套时别删错 ✗）。
    """
    from PyQt5.QtWidgets import QApplication
    from gui.pet_picker import PetPickDialog

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")
    ok, why = petlib.pet_library_state()
    if not ok:
        print("       （跳过：宠物图库还没导出 —— %s）" % why.splitlines()[0])
        return

    d = PetPickDialog([])
    check(d.cand_pet.count() > 0, "左栏一只宠物都没列（图库明明有 ✗）")
    pets = d.cand_pet.ids()
    pid = pets[0]
    d.cand_pet.set_current_id(pid)
    app.processEvents()
    want = [e["id"] for e in petlib.load_pet_equip_index() if pid in (e.get("pets") or [])]
    check(d.cand_eq.ids() == want or sorted(d.cand_eq.ids()) == sorted(want),
          "右栏不是「这只宠物能戴的装备」（该按清单里的 `pets` 过滤 ✓）：%r vs %r"
          % (d.cand_eq.ids()[:5], want[:5]))
    check(d.cand_eq.count() > 0, "这只宠物一件装备都戴不了？不太可能（真源对不上 ✗）")

    eq_id = d.cand_eq.ids()[0]
    d.cand_eq.set_current_id(eq_id)
    app.processEvents()
    check(d._selected_equips() == [eq_id],
          "选中的装备不是 1 件（`ids()` 是全部格子 ✗ 别拿它当选中 ✓）：%r" % (d._selected_equips(),))
    check(d.lbl_preview.pixmap() is not None and not d.lbl_preview.pixmap().isNull(),
          "组合预览没出来（选完该能看到戴着的样子 ✓）")
    d._add()
    got = d.picked_pets()
    check(len(got) == 1 and got[0]["pet"] == pid and got[0]["equips"] == [eq_id],
          "加入的那一套不对：%r" % (got,))
    d._add()
    check(len(d.picked_pets()) == 1, "同一套被重复加了：%r" % (d.picked_pets(),))
    # 同一只宠物、换一件装备 ⇒ **要能加第二套** ✓（戴不同装备是两套外观 ✓）
    if len(want) > 1:
        d.cand_eq.set_current_id(want[1])
        app.processEvents()
        d._add()
        check(len(d.picked_pets()) == 2,
              "同一只宠物配另一件装备加不进去（那不算重复 ✗）：%r" % (d.picked_pets(),))
    # 移出：按那一套删 ✓
    d.lst.set_current_id(pid)
    app.processEvents()
    before = len(d.picked_pets())
    d._remove()
    check(len(d.picked_pets()) == before - 1,
          "「移出选中」没删掉那一套：%r" % (d.picked_pets(),))


def t_card_pet_block():
    """⑥ 卡片那块「标注宠物」（用户 2026-10-04 ✓ 方案 C）：勾选显隐 / 列表 / 阈值 / 分派 / 存盘。

    钉六件（与掉落物那块**逐条对称** ✓ 它就是照那块抄的 ✓）：
      ① 默认藏着（没勾 ⇒ 卡片上不该平白多一大块 ✓）；
      ② 有 `label_pets_on` 与**它自己的**「匹配阈值」`pet_thresh` ✓（用户点名要 ✓）；
      ③ 列表里有项目 ⇒ 两个按钮才亮，灰着要说清为什么 ✓；
      ④ `make_task` 分派到 `tools.label_pets.run_label_pets` ✓ 且**阈值取这一块自己的** ✓
         （兜底链 `pet_thresh` → 总的 `thresh` ✓ 老项目行为不变 ✓）、可视化目录是 `..._pet` ✓；
      ⑤ 勾选 / 阈值 / 列表**都按项目存**（切项目不丢 ✓）；
      ⑥ 源码钉：`_pet_rows` 用 `gui.pet_picker.combo_entry`（**口径只此一处** ✗ 别两处各拼一套 ✓）。
    """
    import shutil
    import tempfile

    from PyQt5.QtWidgets import QApplication
    from gui.project import Project
    from gui.steps.cards import LabelCard

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")
    c = LabelCard()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="petcard_"))
    try:
        check(not c.pet_box.isVisibleTo(c),
              "没勾「标注宠物」就把那一块显示出来了（卡片最下面会平白多一大块 ✗）")
        check("label_pets_on" in c.widgets and "pet_thresh" in c.widgets,
              "卡片上缺「标注宠物」勾或**它自己的**匹配阈值（用户 2026-10-04 点名要 ✗）")
        check(not c.btn_pet_tpl.isEnabled() and "宠物" in c.btn_pet_tpl.toolTip(),
              "列表空着按钮却亮了 / 灰着没说为什么 ✗：%r" % (c.btn_pet_tpl.toolTip(),))
        c.ck_pet.setChecked(True)
        check(c.pet_box.isVisibleTo(c), "勾上「标注宠物」那一块却没出现 ✗")

        p = Project.create(tmp / "proj", name="用例", map_id="105090600")
        c.project = p
        # ⚠ 总阈值要**经卡片载入**（`load_from_project` ✓）—— 直接改项目文件的话，
        #   卡片里那个控件还是默认 0.9 ✗，随后 `sync()` 会把它**写回去** ✗（本轮用例就这么红过 ✓）。
        p.sec("label")["thresh"] = 0.86
        c.load_from_project(p)
        # ⚠ 顺序要紧：`load_from_project` 会**按项目回填列表**（这里项目里还没有 ⇒ 空表 ✓）
        #   ⇒ 手工造的那一套必须在**它之后**放 ✓（不然刚放完就被回填清掉 ✗ 本轮用例又红了一次 ✓）
        c._pets = [{"pet": "5000000", "name": "褐色小猫", "equips": ["01802000"]}]
        c._refresh_pet_list()
        check(abs(float(c.widgets["thresh"][0].value()) - 0.86) < 1e-9,
              "总阈值没经卡片载入（后面那条「别写坏总阈值」就测不出真东西 ✗）：%r"
              % (c.widgets["thresh"][0].value(),))
        check(c.grid_pet.count() == 1 and c.btn_pet_tpl.isEnabled(),
              "列表里有 1 套，格子数/按钮状态不对：%d / %r"
              % (c.grid_pet.count(), c.btn_pet_tpl.isEnabled()))
        c._label_target = "pet"
        c._pet_mode = "template"
        p.sec("label")["thresh"] = 0.86
        fn, params = c.make_task(p)
        check(getattr(fn, "__name__", "") == "run_label_pets",
              "宠物按钮没走到 `tools/label_pets.run_label_pets`（那按钮就是假的 ✗）：%r" % (fn,))
        check(abs(float(params["thresh"]) - 0.86) < 1e-9,
              "没存过 `pet_thresh` 时该用总的那个阈值（老项目行为不变 ✗）：%r" % (params["thresh"],))
        check(str(params["vis_dir"]).endswith("vis_pet"),
              "可视化目录没用单独的 `..._pet`（会和怪物/掉落的图混 ✗）：%r" % (params["vis_dir"],))
        check(params["pets"] and params["pets"][0]["pet"] == "5000000",
              "分派参数里没带上宠物列表：%r" % (params.get("pets"),))
        # 只改这一项 ⇒ 用新的、总的不动 ✓
        c.set_value("pet_thresh", 0.72)
        c.sync(p)
        _fn2, params2 = c.make_task(p)
        check(abs(float(params2["thresh"]) - 0.72) < 1e-9,
              "改了宠物自己的阈值却没生效（那这个参数就是假的 ✗）：%r" % (params2["thresh"],))
        p.save()
        sec = Project.open(tmp / "proj").sec("label")
        check(sec.get("label_pets_on") is True
              and abs(float(sec.get("pet_thresh", -1)) - 0.72) < 1e-9
              and sec.get("pets") == [{"pet": "5000000", "name": "褐色小猫",
                                       "equips": ["01802000"]}],
              "勾选/阈值/列表没按项目存（切项目就丢 ✗）：%r" % ({k: sec.get(k) for k in
                                                ("label_pets_on", "pet_thresh", "pets")},))
        check(abs(float(sec.get("thresh", -1)) - 0.86) < 1e-9,
              "改宠物阈值把**总阈值**也写坏了（不该 ✗）：%r" % (sec.get("thresh"),))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    _csrc = (ROOT / "gui" / "steps" / "cards.py").read_text(encoding="utf-8")
    check("combo_entry" in _csrc and "gui.pet_picker import combo_entry" in _csrc,
          "卡片自己另拼了一套条目（口径该只在 `gui.pet_picker.combo_entry` ✗）")


def t_label_pets_dispatch():
    """⑦ 执行端：**两条都复用既有算法**，且类别号是 **class 5（宠物）**（⛔ 空跑当成功最坏 ✗）。

    用**假算法**把参数截下来钉（同 `selftest_label_drops.t_worker_dispatch` 的做法 ✓）：
      ① 组合外观**现算**过 ⇒ `run_detect` 拿到的 `sprites` = 组合根 ✓、`mobs` = 组合目录名 ✓；
      ② `cls` = `CLASS_PET`（5）✓ —— 写错类别号就是把宠物标成怪 ✗；
      ③ 列表空 / yolo 没权重 ⇒ **抛错**（不静默成功 ✗）；
      ④ yolo 那条走 `mode="pet"` ✓（⇒ 只合并 class 5 ✓ 见 `yolo_augment.MODE_CLASS` ✓）。
    """
    import types

    from perception.classes import CLASS_PET
    from tools import label_pets

    got = {}

    def fake_detect(params, ctx=None):
        got.update(params)
        return {"fake": "detect"}

    real = label_pets.__dict__.get("run_detect")
    import tools.detect_mobs as dm
    old = getattr(dm, "run_detect", None)
    dm.run_detect = fake_detect
    try:
        try:
            label_pets.run_label_pets({"pets": []})
            check(False, "一条宠物都没有却不报错（会「空跑当成功」✗）")
        except ValueError:
            pass
        r = label_pets.run_label_pets({
            "pets": [{"pet": "5000000", "equips": ["01802000"]}],
            "out": "x", "frames": "f", "thresh": 0.77,
        })
        check(r == {"fake": "detect"}, "执行端没返回假算法的结果：%r" % (r,))
        check(got.get("cls") == CLASS_PET,
              "写出去的类别号不是 class 5（宠物 ✗ —— 会让模型把宠物学成怪）：%r" % (got.get("cls"),))
        check(got.get("sprites") == str(petlib.combo_root()),
              "模板根目录不是组合目录（`pet_combo` ✓）：%r" % (got.get("sprites"),))
        check(got.get("mobs") == [petlib.combo_key("5000000", ["01802000"])],
              "模板名不是组合目录名：%r" % (got.get("mobs"),))
        check(abs(float(got.get("thresh", 0)) - 0.77) < 1e-9,
              "匹配阈值没传下去（卡片上设的那个就白设了 ✗）：%r" % (got.get("thresh"),))
        check(got.get("min_side") == label_pets.PET_MIN_SIDE,
              "没走宠物那档「小图闸」（见 `PET_MIN_SIDE` ✗）：%r" % (got.get("min_side"),))
        # yolo：没权重 ⇒ 抛错 ✓
        try:
            label_pets.run_label_pets({"pets": [{"pet": "5000000"}], "mode": "yolo",
                                       "out": "x", "frames": "f", "weights": ""})
            check(False, "YOLO 模式没选权重却不报错 ✗")
        except ValueError:
            pass
    finally:
        if old is None:
            delattr(dm, "run_detect")
        else:
            dm.run_detect = old


def t_pet_buttons_open_frame_picker():
    """⑧ ⭐⭐ 宠物那两个按钮**先开选帧弹窗**（用户 2026-10-04 ✓ 同掉落物那条 ✓ 参考怪物那四个 ✓）。

    钉四件：① 模版匹配 ⇒ 先选帧，**取消 ⇒ 什么都不跑** ✓；② YOLO ⇒ 弹选帧但**不问确认**
    （追加性质 ✓）；③ 选帧结果透传给 `only` ✓；④ 执行端台账按 **"pet"** 记 ✓
    （`tools/label_pets.py` 里传 `"target": "pet"` ✓ —— 不传就成了冒名记进 `mob` ✗）。
    """
    from PyQt5.QtWidgets import QApplication
    from gui import labelio
    from gui.steps.cards import LabelCard

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")
    c = LabelCard()
    c._pets = [{"pet": "5000000", "name": "褐色小猫", "equips": []}]
    c._refresh_pet_buttons()
    check(c.btn_pet_tpl.isEnabled(), "列表里有东西按钮却是灰的 ⇒ 下面几步测不了 ✗")

    calls, emitted = [], []
    c.run_clicked.connect(lambda _w: emitted.append(c._label_target))
    c._confirm_relabel = lambda target: True
    # ① 取消选帧 ⇒ 不跑
    c._pick_frames = lambda target: (calls.append(target), (False, None))[1]
    c.btn_pet_tpl.click()
    check(calls == ["pet"] and not emitted,
          "宠物模版匹配没先开选帧 / 取消后照样跑了 ✗：%r / %r" % (calls, emitted))
    # ② YOLO ⇒ 弹选帧、不确认；③ 透传
    c._confirm_relabel = lambda target: (_ for _ in ()).throw(
        AssertionError("宠物 YOLO 是追加性质 ⇒ 不该问确认（同 YOLO 怪物/玩家 ✗）"))
    c._pick_frames = lambda target: (calls.append(target), (True, ["frame_00003"]))[1]
    emitted.clear()
    c.btn_pet_yolo.click()
    check(calls[-1] == "pet" and c._only.get("pet") == ["frame_00003"] and emitted == ["pet"],
          "宠物 YOLO 的选帧/透传不对：%r / %r / %r"
          % (calls[-1:], c._only.get("pet"), emitted))
    # ④ 台账口径（源码钉 + 行为）
    _lp = (ROOT / "tools" / "label_pets.py").read_text(encoding="utf-8")
    check('"target": "pet"' in _lp,
          "`label_pets` 没传 target=pet ⇒ 宠物跑完会冒名记进「怪物」台账 ✗")
    check(labelio.target_zh("pet") == "宠物" and labelio.target_zh("drop") == "掉落物",
          "target 的文案说法不对（弹窗标题会叫错 ✗）：%r / %r"
          % (labelio.target_zh("pet"), labelio.target_zh("drop")))


def t_pet_task_carries_sprite_scale():
    """⑨ ⭐⭐⭐ **宠物任务必须带上项目的标定尺度**（用户 2026-10-04 ✓ 现场原话：小白雪人
    5000020 "模板匹配标注0检出"）。

    现场实测（同 8 帧、同模板、本项目标定 **1.406** ✓）：
      · `scale=1.0`（老默认 ✗）⇒ **任何阈值**（0.90 / 0.85 / 0.80 / 0.70）都 **0 框** ✗；
      · `scale=1.406` ⇒ 能检出 ✓。
    ⇒ 病根就是这张卡片**一个尺度都没传** ✗（怪那条一直传 `p.get("scale") or 1.12` ✓、
      玩家那条传 `player_scale` ✓），`label_pets` 于是落到默认 1.0 ✓ ⇒ 模板比画面上的宠物
      小 ~40% ✗ ⇒ 一个都匹配不上 ✓。**跟阈值无关** ✗（调低也没用 ✓）。
    ⚠ 顺带钉：没标定过（`scale` 空）⇒ 兜底 **1.12** ✓（= 怪物那条一直用的数 ✓ 别乱改 ✗）。
    """
    import shutil
    import tempfile
    from pathlib import Path

    from PyQt5.QtWidgets import QApplication
    from gui.project import Project
    from gui.steps.cards import LabelCard

    # ⚠⚠ **没它建 QWidget 会原生崩**（`0xC0000005`、且**一句 Python 异常都没有** ✗
    #   —— 本轮踩过：三个新用例全崩、连 PASS/FAIL 都不打印 ✓ 排查代价不小 ✓）。
    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    tmp = Path(tempfile.mkdtemp(prefix="pet_scale_"))
    try:
        p = Project.create(tmp / "proj", name="用例", map_id="105090600")
        p.frames.mkdir(parents=True, exist_ok=True)
        (p.frames / "frame_00000.png").write_bytes(b"x")
        c = LabelCard()
        c._pets = [{"pet": "5000020", "name": "小白雪人", "equips": []}]
        c._label_target = "pet"
        c._pet_mode = "template"

        p.set("scale", 1.406)
        _fn, pr = c.make_task(p)
        check(pr.get("scale") == 1.406,
              "宠物任务没带项目标定尺度（模板会小 40%% ⇒ 0 检出 ✗）：%r" % (pr.get("scale"),))

        p.set("scale", 0)                     # 没标定过
        _fn, pr = c.make_task(p)
        check(pr.get("scale") == 1.12,
              "没标定时没兜底到 1.12（与怪物那条不一致 ✗）：%r" % (pr.get("scale"),))

        # ⭐⭐ **③ 给宠物单独标过 ⇒ 就用那一个**（用户 2026-10-05 ✓ 原话："卡片3标定尺度
        #   也要支持标定掉落物、宠物类"）：上面那两条钉的是**没标时的兜底**（回落总尺度 /
        #   1.12 ✓ 老项目一字不变 ✓），这条钉**标了之后它优先生效** ✓。
        p.set("pet_scale", 2.05)
        _fn, pr = c.make_task(p)
        check(pr.get("scale") == 2.05,
              "③ 给宠物单独标定的尺度没被用上（还在拿总尺度跑 ✗）：%r" % (pr.get("scale"),))
        p.set("pet_scale", 0)                 # 0 = 没标过（回落总尺度 ✓）
        _fn, pr = c.make_task(p)
        check(pr.get("scale") == 1.12,
              "把 pet_scale 清掉后没回落（老项目会崩 ✗）：%r" % (pr.get("scale"),))

        # ⭐⭐ **第二道坎：降采样不能跟着"怪物那个"走**（用户 2026-10-04 ✓ 实测抓到的 ✓）
        #   本项目 `label.downscale = 2` ✓（按怪物 ~100px 设的 ✓），而宠物在画面上才 ~70px
        #   ⇒ 缩一半就糊 ⇒ **0 框** ✗（实测：同 8 帧，`2` ⇒ 0 框、`1` ⇒ 检出 ✓）
        #   ⇒ 这一块必须有**自己的**降采样、默认 1 ✓（**不**回落总降采样 ✗）。
        p.sec("label")["downscale"] = 2            # ⚠ 降采样在 **label 段**里（不是项目顶层 ✗）
        _fn, pr = c.make_task(p)
        check(pr.get("downscale") == 1,
              "宠物任务的降采样跟着怪物那个走了（存的 2 ⇒ 小目标压糊 ⇒ 0 框 ✗）：%r"
              % (pr.get("downscale"),))
        p.sec("label")["pet_downscale"] = 3        # 用户自己调过 ⇒ 照他调的来 ✓
        _fn, pr = c.make_task(p)
        check(pr.get("downscale") == 3,
              "用户自己设的「降采样（宠物）」没被用上 ✗：%r" % (pr.get("downscale"),))
        p.sec("label").pop("pet_downscale", None)

        # ⚠ 源码钉：这三条**共用**同一个 `_sprite_scale` ✓（别再各写一份 ✗）
        src = (ROOT / "gui" / "steps" / "cards.py").read_text(encoding="utf-8")
        bad = [ln for ln in src.splitlines()
               if '"scale": p.get("scale")' in ln and not ln.strip().startswith("#")]
        check(not bad, "有人绕过 `_sprite_scale` 自己写尺度了 ✗：%r" % (bad[:2],))
        # ⚠ 2026-10-05 起这两条**各带自己的 kind**（`drop` / `pet` ✓ 用得上 ③ 单独标定的值 ✓）
        for tag in ('"scale": self._sprite_scale(p, "drop"),',
                    '"scale": self._sprite_scale(p, "pet"),'):
            check(src.count(tag) == 1,
                  "掉落物/宠物那两条没各带自己的 kind（%s 命中 %d 次 ✓ 该 1 次 ✓）"
                  % (tag, src.count(tag)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_pet_entry_shows_real_frame_count():
    """⑩ ⭐ **宠物条目要写清"几帧"**（用户 2026-10-04 ✓ 原话："小白雪人有很多帧，为什么添加后
    写的只有1帧？"✗）。

    实际 `stand0` 有 3 帧 ✓（模板匹配**这 3 帧都会当模板** ✓），可原来 `combo_entry` 不填
    `frames` ✗ ⇒ `icon_grid.normalize_entry` 按默认 1 算 ⇒ 信息窗写「1 张静态图」✗
    （截图里那句 ✓）—— 界面说的和干的事不一致 ✓。
    ⭐ 2026-10-04 口径跟上了**姿态列表**（`petlib.COMBO_STATES` ✓ 站立/呼吸/走路/跳/饿 ✓
      用户要求"四姿态都要能匹配"✓）⇒ 这里**按那个列表数**（别再写死 stand0 的 3 ✗ ——
      写死的话每扩一个姿态这条就得改一次 ✓ 见 `t_pet_combo_states_cover_four` ✓）。
    """
    from PyQt5.QtWidgets import QApplication
    from gui import icon_grid
    from gui.pet_picker import combo_entry

    app = QApplication.instance() or QApplication([])   # ⚠ 同上：QIcon/QPixmap 也要它 ✓
    check(app is not None, "建不起 QApplication")

    e = combo_entry({"id": "5000020", "name": "小白雪人"}, [])
    # ⚠ 按**姿态列表**数（别写死 3 ✗ —— 扩姿态时这里会自动跟着走 ✓ 见上面那段说明 ✓）
    from core import petlib as _pl
    d = _pl.compose_pet("5000020", [])
    _want = sum(len(list(d.glob("%s_*.png" % st))) for st in _pl.COMBO_STATES)
    check(_want > 0 and int(e.get("frames") or 0) == _want,
          "宠物条目帧数不对（该是姿态列表里所有帧 %r ✓）：%r" % (_want, e.get("frames")))
    lines = " ".join(icon_grid.tip_lines(e))
    check("%d 帧" % _want in lines and "静态图" not in lines,
          "信息窗还写着静态图 ✗（用户就是被这句问住的 ✓）：%r" % (lines,))
    # 缩略图要有内容（不能是空图标 ⇒ 界面上就是个黑框 ✗）
    ic = icon_grid.entry_icon(e, 48)
    check(not ic.isNull(), "组合外观的图标是空的（界面上就是那个黑框 ✗）")


def t_pet_name_fixes():
    """⑪ ⭐⭐ **名字手改表**（用户 2026-10-04 ✓ 原话："宠物5000042、5000046名称错了，正确的是
    「花蘑菇仔」，现在是雪娃娃"）。

    现场（拿真 WZ 读了一遍 + 把贴图并排画出来核过 ✓）：客户端 `String.wz/Pet.img` 里
    5000041/42/43/46 **四个 id 全叫「雪娃娃」** ✗，可贴图分别是 雪人 / **橙色蘑菇** / 刺猬 /
    **橙色蘑菇**（42 与 46 的像素指纹一模一样 ⇒ 同一只模型 ✓）⇒ **源数据就错** ✗，不是我们导错 ✓。
    ⇒ 纠正只能放**我们这侧**，而且必须放**本仓库**（`datasets/pets_name_fixes.json` ✓）：
      重导 `dump-pets` 会覆盖 `pets.json` ✗，把名字写那边就白改了 ✓。
    钉五件：① 没表 ⇒ 一字不动 ✓；② 有表 ⇒ 用新名 ✓，且**导出那份 `pets.json` 不许被改** ✓；
    ③ 表坏了 ⇒ 只是不纠正 ✓ **不许抛** ✗；④ `_note` 这类说明键不进映射 ✓；
    ⑤ **真实那份表**对得上（`5000042/46 ⇒ 花蘑菇仔` ✓）—— 哪天谁把表删了、或者重导把名字
      冲掉了，这条会红 ✓。
    """
    import json as _json
    import pathlib as _pathlib

    root = _tmpdir()
    lib = _Lib(root, pets=("5000028",))
    try:
        check(petlib.load_pet_index()[0]["name"] == "进化龙",
              "还没有手改表，名字就变了？%r" % (petlib.load_pet_index(),))
        lib.fixes.write_text(_json.dumps({"_note": "说明键不该进映射", "5000028": "手改的名"},
                                         ensure_ascii=False), encoding="utf-8")
        petlib.clear_pet_caches()
        got = petlib.load_pet_index()
        check(got[0]["name"] == "手改的名",
              "手改表没生效（用户点名要的就是它 ✗）：%r" % (got,))
        check("进化龙" in lib.pet_idx.read_text(encoding="utf-8"),
              "手改表把**导出的** pets.json 也改了 ✗（那份重导会覆盖 ⇒ 不许动 ✓）")
        check(petlib.load_pet_name_fixes().get("_note") is None,
              "`_note` 这种说明键进了映射表（不是 id ✗）")
        lib.fixes.write_text("{ 坏掉的 json", encoding="utf-8")
        petlib.clear_pet_caches()
        check(petlib.load_pet_index()[0]["name"] == "进化龙",
              "手改表坏了就该退回原名（且**不许抛** ✗）")
    finally:
        lib.restore()

    # ⑤ 真实那份：表在、且名单里真的用上了 ✓
    real = _pathlib.Path(petlib.DEFAULT_PET_NAME_FIXES)
    if real.is_file():
        fixes = petlib.load_pet_name_fixes()
        check(fixes.get("5000042") == "花蘑菇仔" and fixes.get("5000046") == "花蘑菇仔",
              "真实手改表里 5000042 / 5000046 该是「花蘑菇仔」✗：%r" % (fixes,))
        idx = {r["id"]: r["name"] for r in petlib.load_pet_index()}
        if "5000042" in idx:
            check(idx["5000042"] == "花蘑菇仔" and idx["5000046"] == "花蘑菇仔",
                  "真实宠物名单没把手改表过一遍（界面还会显示「雪娃娃」✗）：%r / %r"
                  % (idx.get("5000042"), idx.get("5000046")))


def t_combo_multi_states():
    """⑫ ⭐⭐ **多姿态进组合外观**（用户 2026-10-04 ✓ 原话："多姿态进组合外观" ✓）。

    为什么（实测 ✓ 见 SKILL 211）：只做 `stand0` ⇒ 宠物一走路就一个都匹配不上 ✗
    （12 帧里 **1 框 / 8%**）；加上 `stand1` + `move` ⇒ **8 框 / 42%** ✓✓
    ⇒ 默认做 `petlib.COMBO_STATES` 那几个姿态 ✓，管线那边**不用改**
    （`detect_mobs.load_templates` 本来就是 `*.png` 通吃 ✓）。
    钉四件：① 默认把多个姿态的帧都产出来 ✓；② `states=("stand0",)` ⇒ 只做那一个 ✓；
    ③ ⚠ 姿态集合**变小**时，目录里旧姿态的帧必须**被清掉** ✗➡✓（不然它们照样被当模板 ✓）；
    ④ 缓存键**带姿态列表** ✓ ⇒ 先做小的、再要默认的 ⇒ **会重做**（不会错用旧的 ✓）。
    """
    import shutil

    root = _tmpdir()
    lib = _Lib(root, pets=("5000028",))
    try:
        petdir = lib.pet / "5000028"
        for nm, col in (("stand1_0.png", (0, 128, 0)), ("move_0.png", (128, 0, 0))):
            _png(petdir / nm, col)
        out = petlib.compose_pet("5000028")
        names = sorted(p.name for p in pathlib.Path(out).glob("*.png"))
        check(any(n.startswith("stand0_") for n in names), "没产出 stand0 帧：%r" % (names,))
        check(any(n.startswith("stand1_") for n in names),
              "**多姿态没生效**（缺 stand1 ✓ 用户点名要的就是它 ✗）：%r" % (names,))
        check(any(n.startswith("move_") for n in names), "缺 move 帧：%r" % (names,))
        mj = json.loads((pathlib.Path(out) / "meta.json").read_text(encoding="utf-8"))
        check(set(mj.get("states") or []) >= {"stand0", "stand1", "move"},
              "meta.json 没记清做了哪些姿态（界面要靠它回读 ✓）：%r" % (mj,))

        # ② 只做 stand0 ⇒ ③ 旧姿态的帧要被清掉
        out2 = petlib.compose_pet("5000028", states=("stand0",))
        names2 = sorted(p.name for p in pathlib.Path(out2).glob("*.png"))
        check(names2 and all(n.startswith("stand0_") for n in names2),
              "`states=('stand0',)` 之后目录里还留着别的姿态 ✗（它们照样会被当模板 ✓）：%r"
              % (names2,))

        # ④ 再要默认 ⇒ 缓存键不同 ⇒ 该重做（把姿态补回来 ✓）
        out3 = petlib.compose_pet("5000028")
        names3 = sorted(p.name for p in pathlib.Path(out3).glob("*.png"))
        check(any(n.startswith("move_") for n in names3),
              "切回默认姿态时被**旧缓存**挡住了（不会重做 ⇒ 永远只有 stand0 ✗）：%r" % (names3,))
    finally:
        lib.restore()
        shutil.rmtree(root, ignore_errors=True)


def t_pet_del_needs_selection():
    """⑬ ⭐⭐ **「移出选中」**：选中才亮、且认得出是**哪一套**（用户 2026-10-04 ✓ 原话：
    "现在选中已添加的宠物，移出选中按钮不能交互" ✓）。

    挖下去是**两个**毛病（都在这一条上翻出来 ✓）：
      ① ⛔ **选中那一刻没人重算按钮** ✗：`_refresh_pet_buttons` 里那条"当前有选中项才亮"的
         判据 ✓ 只在「添加…/移出选中/忙闲」时才跑 ⇒ 你在格子里点中一只时按钮**还是灰的** ✓
         点了毫无反应 ✓（离屏复现：`current_id` 已经对了 ✓ 而 `isEnabled()` 还是 False ✗）
         ⇒ 修：把格子的 `itemSelectionChanged` 接到 `_refresh_pet_buttons` ✓。
      ② ⛔ **格子把 `pet`/`equips` 丢了** ✗：`icon_grid.normalize_entry` 是**重拼**一个新 dict ✓
         ⇒ `combo_entry` 带来的这两个字段被吃掉 ✓ ⇒ `_pet_del` 认不出"哪一套" ✓
         ⇒ **戴了装备的那套永远删不掉** ✗（没装备那套因为 `want=[]` 恰好对得上才删得掉 ✓
         —— 所以这个 bug 藏了很久 ✓）。修：原条目里多出来的键**一并带着** ✓（本函数算出来的
         那些仍是权威值 ✓ 不许被覆盖 ✓）。
    用例把两件都钉住：按钮的亮/灭跟着选中走 ✓；**同一只宠物的两套装备**只删掉选中那套 ✓。
    """
    import shutil

    root = _tmpdir()
    lib = _Lib(root, pets=("5000028",), equips=(("01802000", ("5000028",)),))
    try:
        from PyQt5.QtWidgets import QApplication
        from gui.steps.cards import LabelCard

        app = QApplication.instance() or QApplication([])
        check(app is not None, "建不起 QApplication")

        c = LabelCard()
        c._pets = [{"pet": "5000028", "name": "进化龙", "equips": []},
                   {"pet": "5000028", "name": "进化龙", "equips": ["01802000"]}]
        c._refresh_pet_list()
        c._refresh_pet_buttons()
        check(not c.btn_pet_del.isEnabled(),
              "什么都没选中 ⇒ 按钮该是灰的 ✗（不然点了没反应更费解 ✓）")

        c.grid_pet.setCurrentRow(1)                     # 选中"带装备"那套
        check(c.btn_pet_del.isEnabled(),
              "**选中之后按钮还是灰的** ✗（用户报的就是这个：点了不能交互 ✓）"
              " —— 检查 `itemSelectionChanged` 有没有接到 `_refresh_pet_buttons` ✓")
        e = c.grid_pet.current_entry() or {}
        check(str(e.get("pet") or "") == "5000028" and list(e.get("equips") or []) == ["01802000"],
              "格子里拿回的选中项**丢了 pet/equips** ✗（那就认不出是哪一套 ✓）：%r" % (e,))

        c.btn_pet_del.click()
        check([d.get("equips") for d in c._pets] == [[]],
              "点「移出选中」没删对：该只删掉**带装备那套** ✓（同一只的另一套要留着 ✓）：%r"
              % (c._pets,))
        check(not c.btn_pet_del.isEnabled(),
              "删完没选中项了 ⇒ 按钮该回到灰的 ✓")
    finally:
        lib.restore()
        shutil.rmtree(root, ignore_errors=True)


def t_pet_combo_states_cover_four():
    """① ⭐⭐ **站立 / 走路 / 跳 / 饿 都要进匹配模板**（用户 2026-10-04 ✓ 原话："宠物的匹配站立
    stand、走路 move、跳 jump、饿 hungry 都需要是匹配的对象"）。

    为什么：只做 `stand0` 时宠物一走路就匹配不上（实测 1 框/8% ✗，见 `petlib.COMBO_STATES` ✓）
    ⇒ 姿态列表就是「**能被匹配到的姿态**」这份名单 ✓；`detect_mobs.load_templates` 拿的是
    `sorted(d.glob("*.png"))` ✓ ⇒ 组合多一个姿态 ⇒ 模板多几张 ✓（执行端不用改 ✓）。
    钉三件：① 常量覆盖四姿态 ✓；② `combo_entry` 的 `frames` 数**所有姿态** ✓（原来只数
    `stand0_*` ✗ ⇒ 界面少报 ✓）；③ 信息窗有「模板姿态」那行 ✓。
    """
    from PyQt5.QtWidgets import QApplication

    from core import petlib
    from gui import icon_grid
    from gui.pet_picker import combo_entry

    QApplication.instance() or QApplication([])
    miss = [s for s in ("stand0", "move", "jump", "hungry") if s not in petlib.COMBO_STATES]
    check(not miss, "组合姿态少了这几套（用户点名要的 ✓）：%r ⇒ COMBO_STATES=%r"
                    % (miss, tuple(petlib.COMBO_STATES)))
    e = combo_entry({"id": "5000020", "name": "小白雪人"}, [])
    d = petlib.compose_pet("5000020", [])
    real = sum(len(list(d.glob("%s_*.png" % st))) for st in petlib.COMBO_STATES)
    check(int(e["frames"]) == real and real > 0,
          "`frames` 没按姿态列表数（界面会少报 ✗）：报 %r / 实际 %r" % (e.get("frames"), real))
    check(all(w in str(e.get("states")) for w in ("站立", "走路", "跳", "饿")),
          "条目里没带「覆盖了哪几个姿态」的中文短语 ✗：%r" % (e.get("states"),))
    check(any("模板姿态" in l for l in icon_grid.tip_lines(e)),
          "信息窗没写「模板姿态」（用户那句疑问的正解 ✗）")


def t_pet_tooltip_has_icon():
    """③ ⭐⭐ **指着宠物时信息弹窗要有图标**（用户 2026-10-04 ✓ 原话："指着宠物的时候信息弹窗
    没有 icon"）。

    病根：`IconTip.set_entry` 原来自己拼 `wzexport.drop_icon_path(e["id"])` ✗ —— 宠物 id 在
    **掉落物图库**里当然查不到 ⇒ 68×68 那个格子一直空着 ✓。修法：走唯一出口
    `icon_grid.entry_pixmap`（`entry["icon"]` 优先 ✓ = 组合图 ✓）。
    钉两件：① 宠物 ⇒ 有图标 ✓；② **图库里没有的 id** ⇒ 仍旧空 ✓（别把"没有图"也画成有 ✗）。
    """
    from PyQt5.QtWidgets import QApplication

    from gui import icon_grid
    from gui.pet_picker import combo_entry

    app = QApplication.instance() or QApplication([])
    tip = icon_grid.IconTip()
    tip.set_entry(combo_entry({"id": "5000020", "name": "小白雪人"}, []))
    pm = tip.lbl_icon.pixmap()
    check(pm is not None and not pm.isNull(),
          "宠物的信息弹窗里没有图标（用户报的就是这个 ✗）")
    check(icon_grid.entry_pixmap({"id": "99999999"}).isNull(),
          "图库里没有的 id 不该画出图来 ✗")
    tip.deleteLater()
    app.processEvents()


def t_export_missing_frames():
    """⑭ ⭐⭐⭐ **缺图就当场补导**（用户 2026-10-05 ✓ 原话："如果没有导出，在添加的带装备的宠物、
    关闭弹窗后，在数据集工作台显式读条导出"）。

    起因（同一轮报的现场 ✓）：给要标注的宠物配上装备后，双击格子在弹窗里看**要标注的帧**
      ⇒ **除了 `stand0`，其它姿态全都没有装备** ✗（鳄鱼潭1 / 花蘑菇仔 ✓）。
    真因（现场实测 ✓）：装备图库 **52 件全只有 `stand0`** ✗（本体有 7 个姿态 ✓）—— 源头是导出器
      `dump-petequips` 的 `--states` **默认 `stand0`** ✓（`WzProbe/PetDump.cs` ✓）；而组合是
      **按姿态分别叠**的 ✓ ⇒ 非站立姿态就没有装备可叠 ✓（界面照实画了 ✓ 界面没错 ✓）。
    ⇒ 处置：加完宠物**关窗之后**，工作台**显式读条**把这几个姿态补导出来 ✓（这条用例钉它 ✓）。

    钉四层（判据 / 命令 / 任务 / 界面钩子 ✓）：
      ① `petlib.combo_export_plan` 只点**真缺**的：装备只有 stand0 ⇒ 只点名装备那条、姿态**不含**
         `stand0` ✓；本体也缺 ⇒ 另加一条 `dump-pets` ✓；都齐 ⇒ **空表**（不打扰人 ✓）；
         ⭐ 同一件装备多只宠物各缺一个姿态 ⇒ **取并集跑一趟** ✓（按宠物拆成两趟是白跑 ✓）；
      ② `wzexport.pet_export_cmd` 的 argv 逐字对 ✓ —— ⚠ **本体那条不许带 `--states`** ✗
         （`PetDump.RunPets` 一趟全导 ✓ 传了语义就变了 ✓）；
      ③ `wzexport.pet_export_task` 的形状与 `card.make_task` 一致（`(fn, params)` ✓）且配置**冻进**
         params ✓（子线程里不再读配置 ✓）；
      ④ 卡片钩子：`_pet_add` 里**真调了** `_export_missing_pet_frames` ✓；有工作台 ⇒ 走
         `run_export`（读条 + 日志 + 取消 ✓）；没工作台 / 工作台正忙 ⇒ **写一句实话** ✓
         （不许静默 ✗）；导完 ⇒ **重刷列表** ✓（图标/预览跟着对 ✓）。
    """
    import shutil
    from types import SimpleNamespace

    pid, eq = "5000042", "01802054"
    pets = [{"pet": pid, "name": "花蘑菇仔", "equips": [eq]}]
    root = _tmpdir()
    lib = _Lib(root, pets=(pid,), equips=((eq, (pid,)),))
    try:
        from gui.steps.cards import LabelCard

        # ---------------- ① 判据：只点真缺的，姿态不含 stand0 ----------------
        plan = petlib.combo_export_plan(pets)
        check([s["cmd"] for s in plan] == ["dump-pets", "dump-petequips"],
              "本体（假图库只有 stand0）+ 装备都缺 ⇒ 该有 **两条**（本体、装备）✓：%r" % (plan,))
        check(plan[0]["ids"] == [pid] and plan[0]["states"] == [],
              "本体那条：id 要指名 ✓、`states` 要**空** ✓（导出器一趟全导 ✓）：%r" % (plan[0],))
        check(plan[1]["ids"] == [eq],
              "装备那条没点名这件装备 ✓：%r" % (plan[1],))
        check(plan[1]["states"] == ["stand1", "move", "jump", "hungry"],
              "装备那条该点名**除 stand0 之外**的姿态、顺序照 `COMBO_STATES` ✓：%r" % (plan[1],))

        # ①b 本体补齐 ⇒ 只剩装备那条（本体不许再被点名 ✓ 不然每加一次都要重导本体 ✗）
        for st in ("stand0", "stand1", "move", "jump", "hungry"):
            _png(lib.pet / pid / ("%s_0.png" % st), (0, 0, 255))
        plan = petlib.combo_export_plan(pets)
        check(len(plan) == 1 and plan[0]["cmd"] == "dump-petequips",
              "本体齐了 ⇒ 只该剩装备那条 ✓：%r" % (plan,))

        # ---------------- ④ 卡片钩子（趁装备还缺着 ✓）----------------
        calls, notes, refreshed = [], [], []

        class _Win:
            def __init__(self, ret=True):
                self.ret = ret

            def run_export(self, fn, params, title, on_done=None):
                calls.append({"fn": fn, "params": params, "title": title, "on_done": on_done})
                return self.ret

        def _fake(win):
            # ⚠ 收尾回调也要在（钩子会把它交给工作台 ✓）；这里替成 no-op（真那条在下面单独测 ✓）
            ns = SimpleNamespace(_pets=list(pets), window=lambda: win,
                                 _pet_note=notes.append,
                                 _refresh_pet_list=lambda: refreshed.append(1))
            ns._on_pet_export_done = lambda ok, summary: None
            return ns

        win = _Win()
        LabelCard._export_missing_pet_frames(_fake(win))
        check(len(calls) == 1, "缺图时该**起一次补导**（走工作台读条 ✓）—— 实得 %d 次" % len(calls))
        check(calls[0]["fn"] is wzexport.run_pet_export,
              "补导该跑 `wzexport.run_pet_export` ✓：%r" % (calls[0]["fn"],))
        check([s["cmd"] for s in calls[0]["params"].get("plan") or []] == ["dump-petequips"],
              "params 里要带上**判据给的计划** ✓（界面不自己拼 ✗）：%r" % (calls[0]["params"],))
        check("1" in str(calls[0]["title"]),
              "标题该说清**补几项**（工作台日志看得到 ✓）：%r" % (calls[0]["title"],))
        check(calls[0]["on_done"] is not None,
              "没给收尾回调 ⇒ 导完没人重刷列表（图标/预览还是旧的 ✗）")

        # 导完 ⇒ 重刷列表 + 说一句（先刷再写 ⇒ 那句话不会被 `_refresh_pet_state` 冲掉 ✓）
        _fake_ns = _fake(_Win())
        refreshed.clear()
        notes.clear()
        LabelCard._on_pet_export_done(_fake_ns, True, "补导 1 项")
        check(refreshed and notes and "已补导" in notes[-1],
              "补导成功 ⇒ 该**重刷列表**并说清结果 ✓：刷新=%r 状态=%r" % (refreshed, notes))

        # 工作台正忙（起不来）⇒ 一句实话（**不许静默** ✗）
        notes.clear()
        LabelCard._export_missing_pet_frames(_fake(_Win(ret=False)))
        check(notes and "再点一次" in notes[-1],
              "工作台正忙时该说清「等它跑完再点一次」✓（静默 ⇒ 人以为补过了 ✗）：%r" % (notes,))

        # 没有工作台（用例 / 别处嵌的卡片）⇒ 也要说清（点名缺几项 ✓）
        notes.clear()
        LabelCard._export_missing_pet_frames(_fake(SimpleNamespace()))
        check(notes and "没有工作台" in notes[-1],
              "没有工作台时该如实说（别静默 ✗）：%r" % (notes,))

        # `_pet_add` 里**真接上**了（防哪天又拆掉：那这条功能就等于没有 ✓）
        # ⚠⚠ 判据必须查 **AST 里的真调用**，不能字符串匹配 ✗ —— 本轮反向验证就是被它放过去的：
        #   把调用注释掉（`pass  # self._export_missing_pet_frames()`）之后，**注释里那串字还在**
        #   ⇒ 字符串匹配照样"有" ✓ ⇒ 用例没红 ✗（功能确实断了 ✓）。注释/说明文不是代码 ✓。
        import ast as _ast

        _src_txt = (ROOT / "gui" / "steps" / "cards.py").read_text(encoding="utf-8")
        _add_fn = next(n for n in _ast.walk(_ast.parse(_src_txt))
                       if isinstance(n, _ast.FunctionDef) and n.name == "_pet_add")
        _hooked = [c for c in _ast.walk(_add_fn) if isinstance(c, _ast.Call)
                   and getattr(c.func, "attr", None) == "_export_missing_pet_frames"]
        check(bool(_hooked),
              "`_pet_add` 里没**真的**调 `_export_missing_pet_frames` ⇒ **关窗之后不会补导** ✗"
              "（⚠ 查 AST，别查字符串：注释里写着那句也算「有」✗）")

        # ---------------- ①c 图齐了 ⇒ 空表 + 不打扰人 ----------------
        for st in ("stand1", "move", "jump", "hungry"):
            _png(lib.eq / eq / pid / ("%s_0.png" % st), (255, 0, 0), size=(1, 1))
        check(petlib.combo_export_plan(pets) == [],
              "图都齐了就该是**空表** ✓（不然每次加宠物都弹一次读条 ✗）")
        calls, notes = [], []
        LabelCard._export_missing_pet_frames(_fake(_Win()))
        check(not calls and not notes,
              "图齐时**什么都不做**（不起任务 ✓ 也不写状态 ✓）：%r / %r" % (calls, notes))
    finally:
        lib.restore()
        shutil.rmtree(root, ignore_errors=True)

    # ---------------- ①d 并集口径：一件装备、两只宠物各缺一个姿态 ⇒ 一趟跑完 ----------------
    root2 = _tmpdir()
    lib2 = _Lib(root2, pets=("5000042", "5000046"),
                equips=((eq, ("5000042", "5000046")),))
    try:
        for p in ("5000042", "5000046"):
            for st in ("stand0", "stand1", "move", "jump", "hungry"):
                _png(lib2.pet / p / ("%s_0.png" % st), (0, 0, 255))
        _png(lib2.eq / eq / "5000042" / "stand1_0.png", (255, 0, 0), size=(1, 1))
        _png(lib2.eq / eq / "5000042" / "move_0.png", (255, 0, 0), size=(1, 1))
        _png(lib2.eq / eq / "5000046" / "stand1_0.png", (255, 0, 0), size=(1, 1))
        _png(lib2.eq / eq / "5000046" / "jump_0.png", (255, 0, 0), size=(1, 1))
        p2 = petlib.combo_export_plan([{"pet": "5000042", "equips": [eq]},
                                       {"pet": "5000046", "equips": [eq]}])
        check(len(p2) == 1 and p2[0]["ids"] == [eq],
              "同一件装备该**一趟**跑完（按宠物拆两趟是白跑 ✓）：%r" % (p2,))
        check(p2[0]["states"] == ["move", "jump", "hungry"],
              "并集该把两只宠物缺的姿态合起来（都已有 stand1 ⇒ 不含 stand1 ✓）、"
              "且顺序照 `COMBO_STATES` ✓：%r" % (p2[0],))
    finally:
        lib2.restore()
        shutil.rmtree(root2, ignore_errors=True)

    # ---------------- ② 命令 argv（本体**不许**带 `--states` ✗）----------------
    _body = wzexport.pet_export_cmd("X.exe", {"cmd": "dump-pets", "ids": [pid], "states": []},
                                    "WZ", "OUT")
    check(_body == ["X.exe", "dump-pets", "WZ", "OUT", "--only", pid],
          "本体那条 argv 不对（⚠ 不许带 `--states` ✗）：%r" % (_body,))
    _eqc = wzexport.pet_export_cmd("X.exe",
                                   {"cmd": "dump-petequips", "ids": [eq],
                                    "states": ["stand1", "move", "jump", "hungry"]},
                                   "WZ", "OUT")
    check(_eqc == ["X.exe", "dump-petequips", "WZ", "OUT", "--only", eq,
                   "--states", "stand1,move,jump,hungry"],
          "装备那条 argv 不对：%r" % (_eqc,))

    # ---------------- ③ 任务形状：与 `card.make_task` 一致 + 配置冻进 params ----------------
    _src = [{"cmd": "dump-petequips", "ids": [eq], "states": ["move"]}]
    fn, params = wzexport.pet_export_task(_src)
    check(fn is wzexport.run_pet_export and isinstance(params, dict),
          "`pet_export_task` 该回 `(run_pet_export, params)`（工作台按这个形状起任务 ✓）")
    check("plan" in params and "exe" in params and "wz_dir" in params,
          "params 里该把计划 + exe + wz_dir **冻好**（子线程里不再读配置 ✓）：%r" % (sorted(params),))
    _src[0]["ids"].append("999")               # 调用方回头改自己那份
    _src[0]["states"].append("jump")
    check(params["plan"][0]["ids"] == [eq] and params["plan"][0]["states"] == ["move"],
          "计划只做了**浅拷** ⇒ 调用方改自己那份就改了**另起线程里正跑着的那份** ✗：%r"
          % (params["plan"],))


def main():
    tests = [
    ("① 宠物组合姿态覆盖 站立/走路/跳/饿（用户 2026-10-04）",
     t_pet_combo_states_cover_four),
    ("③ 宠物信息弹窗必须带图标（用户 2026-10-04）",
     t_pet_tooltip_has_icon),
        ("图库现状：全缺/缺装备/齐全，三种都要如实说（点名 WzProbe 该加的两个通道）", t_library_state),
        ("帧与 origin：只认 <state>_<n>、按帧号排序、meta 坏了退回中心原点", t_frames_and_meta),
        ("⭐⭐ 组合外观：按 origin 对齐叠成 2×3 / origin(1,2) / 装备落在该落的格子；缺帧取模；缓存与失效", t_compose),
        ("⭐⭐ 真图库走一遍（没有就明确跳过）：清单完整、能戴关系在、组合真叠上装备", t_real_library_and_compose),
        ("⭐ 弹窗流程：搜索宠物 → 只列它能戴的装备 → 选 1 件只加 1 件 → 同只可配多套 → 按套移出",
         t_pet_picker_flow),
        ("卡片「标注宠物」块：勾选显隐 / 自己的匹配阈值 / 分派到 run_label_pets / 按项目存",
         t_card_pet_block),
        ("执行端：现算组合外观当模板、写 **class 5（宠物）**、空列表或没权重要抛错（不空跑）",
         t_label_pets_dispatch),
        ("⭐⭐ 宠物按钮**先开选帧弹窗**（同怪物那四个）：取消⇒不跑、YOLO 不问确认、"
         "结果透传；台账按 `pet` 记",
         t_pet_buttons_open_frame_picker),
        ("⭐⭐⭐ 宠物任务**必须带项目的标定尺度**（不传 ⇒ 默认 1.0 ⇒ 模板小 40% "
         "⇒ 0 检出；没标定兜底 1.12）",
         t_pet_task_carries_sprite_scale),
        ("⭐ 宠物条目写清**真实帧数**（stand0 有 3 帧 ⇒ 不许再写「1 张静态图」✗）",
         t_pet_entry_shows_real_frame_count),
        ("⭐⭐ 宠物名**手改表**：客户端 String.wz 名字重了/错了（5000042/46 明明"
         "是花蘑菇仔却叫雪娃娃）⇒ 按 id 纠正、不动导出那份、坏了也不抛",
         t_pet_name_fixes),
        ("⭐⭐ **多姿态进组合外观**（stand0+stand1+move ⇒ 实测 8% → 42%）：默认做多个"
         "姿态、`states=` 能收窄、收窄后会清掉旧姿态的帧、缓存键带姿态列表",
         t_combo_multi_states),
        ("⭐⭐ **「移出选中」**：选中才亮（选中变化要重算按钮）+ 认得出是**哪一套装备**"
         "（格子不许把 pet/equips 丢掉）",
         t_pet_del_needs_selection),
        ("⭐⭐⭐ **缺图就当场补导**（用户 2026-10-05）：只点真缺的姿态（含 stand0 就白导 ✓）、"
         "本体那条不带 `--states`、同一件装备取并集一趟、关窗后走**工作台读条**、"
         "忙/没工作台都要如实说、导完重刷列表",
         t_export_missing_frames),
    ]
    ok = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as e:                                    # noqa: BLE001
            print("[FAIL] %s\n       %s: %s" % (name, type(e).__name__, e))
        else:
            print("[ OK ] %s" % name)
            ok += 1
    print("\n%d/%d 通过" % (ok, len(tests)))
    return 0 if ok == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
