"""**自检索引**：改动落在哪些文件 ⇒ 只跑**相关**的那几个套件 ✓（用户 2026-10-17 要求）。

为什么要有它（用户原话）：
    「**整理自检和文档，以后的开发，无关模块的自检、读文档就不要进行了**
      如果功能有耦合，优化代码结构」✓

⚠ 这是**规矩**，不是建议 ✗：
  · 改完**先** `python -m tools.selftest_index --changed`（会列出该跑哪些 ✓）；
  · **只**跑列出来的那些 ✓（全量只在"跨层重构 / 改共享底座"时才跑 ✓）；
  · 文档同理：**只读** `AFFECTS` 里那几个套件 + 本次改到的模块的文档 ✗（别通读 ✓）。

跑法：
    python -m tools.selftest_index                      # 打印「源 → 套件」对照表
    python -m tools.selftest_index decision/route.py    # 这些文件该跑哪些
    python -m tools.selftest_index --changed            # 按 git 改动（含未跟踪）算
    python -m tools.selftest_index --changed --run      # 直接跑（子进程，一个套件一个进程）

⚠ 改了这个映射表 ⇒ 顺手在 `docs/自检索引.md` 里同步一句 ✗（那边是给人看的 ✓）。
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: ⭐ **每个套件"管什么"**（写给人看的那一层 ✓ 一句话 ✓ 别写成代码注释 ✗）。
SUITES = {
    "selftest_decision": "决策层 / 按键层：路线、爬绳、鼠标、断线重连状态机",
    "selftest_arch": "架构边界（唯一写者 / 分层 / 别长出第二条链）",
    "selftest_review": "代码质检（←/→ 切帧、评审台）",
    "selftest_minimap": "小地图：定位、黄点、框选比例、窗口绑定",
    "selftest_screen_state": "界面状态（测谎弹窗）+ live_thread 的判定/录屏逻辑",
    "selftest_live_panel": "实时预览面板 + 管路闸（FrameCtx / HeavyGate）",
    "selftest_main_window": "主窗口页签开合",
    "selftest_deploy": "部署台（参数表 / 命令行 / 卡片界面）",
    "selftest_stream": "【B 机】一键自测收流链路（要真流 ✓ 不是断言套件）",
    "selftest_zones": "寻路数据层（footprint 集合 / 边 / 路径点）",
    "selftest_route_cfg": "「路线识别」配置按地图 id 存",
    "selftest_mmap_rate": "小地图高帧率更新那条链",
    "selftest_mmap_regions": "小地图推流区域按地图 id 存 + 运行中换图",
    "selftest_tracker": "怪追踪 / 玩家追踪",
    "selftest_anchor_track": "锚定追踪",
    "selftest_kbd_release": "把键松干净那条路",
    "selftest_behavior": "行为打点（一行一个事件、人要读得懂）",
    "selftest_config_sync": "双机配置一致性",
    "selftest_machineload": "本机负载清点",
    "selftest_winperf": "性能保活（实测 sleep 精度）",
    "selftest_region": "框选控件（放大镜 / 坐标换算）",
    "selftest_seq_editor": "行为编辑器保存 / 加载预设",
    "selftest_zone_editor": "集合编辑器（拾取 / 框选 / 选择集）",
    "selftest_label_drops": "掉落物自动标注",
    "selftest_pets": "宠物图库 + 组合外观模板",
    "selftest_metrics": "训练建议（数字必须来自 run.json）",
    "selftest_yolo_wb": "yolo 工作台（纯逻辑 + 离屏）",
    "selftest_train_guard": "训练链路的两个守卫",
    "selftest_train_report": "查看报告弹窗",
    "selftest_push_presets": "推流账（命令 / 指标 / 两机对账）",
    "selftest_probe_tune": "探针手动微调（正几何解对、偏几何判出）",
    "selftest_engage": "输出收益口径（纯函数）",
    "selftest_link": "指令通道能不能到 Pro Micro（要硬件 ✓）",
    "selftest_sweep_link": "握手通道（把「对上」钉死）",
    "selftest_lie_controller": "M2c 鼠标闭环控制律（模拟光标）",
    "selftest_lie_demo": "测谎演示窗口（离屏）",
    "selftest_lie_group": "A 层 lie_group（合成场景）",
    "selftest_lie_motion": "lie_motion（YOLO 候选 + 运动不合群）",
    "selftest_lie_registry": "假目标登记表",
    "selftest_lie_tracker": "lie_tracker 合成序列",
    "selftest_live_lie": "实时测谎测试窗（离屏）",
    "selftest_vt_sdk": "自动测谎 SDK 薄封装",
}

#: ⭐⭐ **改动 → 该跑哪些**（**先看具体文件，再看目录** ✓ 命中即止 ✓）。
#: ⚠ 加新文件时**优先复用目录那一条** ✗（只有"它跟别处不一样"才单列 ✓）。
AFFECTS = {
    # ---- 决策层 ----
    "decision/route.py": ("selftest_decision", "selftest_zones"),
    "decision/route_cfg.py": ("selftest_decision", "selftest_route_cfg"),
    "execution/mouse_aim.py": ("selftest_decision",),
    "decision/reconnect.py": ("selftest_decision", "selftest_live_panel"),
    "decision/agent.py": ("selftest_decision", "selftest_arch"),
    "decision/engage.py": ("selftest_engage", "selftest_decision"),
    "decision/": ("selftest_decision",),
    # ---- 感知层 ----
    "perception/minimap.py": ("selftest_minimap", "selftest_screen_state"),
    "perception/screen/screen_state.py": ("selftest_screen_state",),
    "perception/screen/ui_state.py": ("selftest_screen_state",),
    "perception/vt_sdk.py": ("selftest_vt_sdk", "selftest_screen_state"),
    "liedetect/lie_group.py": ("selftest_lie_group",),
    "liedetect/lie_motion.py": ("selftest_lie_motion",),
    "liedetect/lie_registry.py": ("selftest_lie_registry",),
    "liedetect/lie_tracker.py": ("selftest_lie_tracker",),
    "perception/track/tracker.py": ("selftest_tracker",),
    "perception/world/anchor_track.py": ("selftest_anchor_track",),
    "perception/": ("selftest_minimap",),
    # ⚠ 测谎 SDK 是**外挂目录**（`visual_tracking_sdk_20260920/liescc` ✓）——
    #   改它就跑 `selftest_vt_sdk` ✓（2026-10-17 在这儿改过双锚点配对的右边容差 ✓）
    "visual_tracking_sdk_20260920/": ("selftest_vt_sdk",),
    # ---- 界面 ----
    "gui/live_thread.py": ("selftest_screen_state", "selftest_live_panel"),
    "gui/live_panel.py": ("selftest_live_panel", "selftest_screen_state"),
    "gui/live_lie.py": ("selftest_live_lie",),
    "gui/route_panel.py": ("selftest_decision", "selftest_route_cfg"),
    "gui/region_selector.py": ("selftest_region",),
    "gui/main_window.py": ("selftest_main_window",),
    "gui/seq_editor.py": ("selftest_seq_editor",),
    "gui/zone_editor.py": ("selftest_zone_editor",),
    "gui/yolo_panel.py": ("selftest_yolo_wb",),
    "gui/": ("selftest_live_panel", "selftest_main_window"),
    # ---- 底座 ----
    "core/wincap.py": ("selftest_minimap",),
    "core/behavior.py": ("selftest_behavior",),
    "core/config.py": ("selftest_config_sync", "selftest_live_panel"),
    "core/zones.py": ("selftest_zones",),
    "core/perf.py": ("selftest_winperf", "selftest_live_panel"),
    # 会话判定（硬件 HID 只进物理控制台会话 ✓ 见 `core/session.py`）：界面靠它提示 ✓
    "core/session.py": ("selftest_live_panel",),
    "runtime/pipeline.py": ("selftest_live_panel",),
    "core/": ("selftest_review",),
    # ---- 共享底座（改了就得多跑几套 ✓ 但**不是全部** ✗）----
    "runtime/": ("selftest_live_panel", "selftest_arch"),
    # ---- 部署台 / 配置 / 工具 ----
    "deploy/": ("selftest_deploy",),
    "config/": ("selftest_config_sync",),
    "tools/probe_tune.py": ("selftest_probe_tune",),
    "tools/lie_demo.py": ("selftest_lie_demo",),
    "tools/live_lie.py": ("selftest_live_lie",),
    "tools/selftest_index.py": (),          # 自检工具自己 ⇒ 不触发别的 ✓
    "tools/": (),                           # 别的工具脚本 ⇒ 不触发 ✓
}

#: ⭐ **改了这些就必须跑全量**（跨层 / 共享契约 ✓ 用户口径："有耦合就优化结构"✓）——
#:   数量故意很少 ✓：**加不进来就别加** ✗（否则等于回到"每次都跑全量"✓）。
ALWAYS_ALL = (
    "core/config.py",       # 所有人读配置
    "core/behavior.py",     # 打点 format 一改，多个套件按字面断言
)


def _norm(p):
    """路径归一：`\\`→`/`、去掉 `./` 与仓库根前缀 ✓。"""
    s = str(p).replace("\\", "/")
    for pre in ("./", str(ROOT).replace("\\", "/") + "/"):
        if s.startswith(pre):
            s = s[len(pre):]
    return s


def suites_for(paths):
    """这些文件 ⇒ `(套件列表, 是否必须全量, 说明行)` ✓（去重、保持**稳定顺序** ✓）。"""
    got, notes = [], []
    all_flag = False
    for p in paths:
        n = _norm(p)
        if n.endswith(".md") or n.startswith("docs/"):
            continue                    # 文档改动不触发套件 ✓
        if n in ALWAYS_ALL:
            all_flag = True
            notes.append("%s ⇒ **共享契约** ⇒ 全量 ✓" % n)
            continue
        hit = None
        for key in sorted(AFFECTS, key=len, reverse=True):   # 具体文件先于目录 ✓
            if n == key or (key.endswith("/") and n.startswith(key)):
                hit = key
                break
        if hit is None:
            # ⚠ **没登记 ≠ 全量** ✗（那就又回到"每次都跑全量"了 ✓ 用户正是不想要这个 ✓）
            #   ⇒ 只**提醒**一句、不触发 ✓；真是共享契约 ⇒ 加进 `ALWAYS_ALL` ✓。
            notes.append("%s ⇒ **没登记** ⇒ 本次不触发（顺手加进 AFFECTS ✓）" % n)
            continue
        for s in AFFECTS[hit]:
            if s not in got:
                got.append(s)
        notes.append("%s ⇒ %s" % (n, ", ".join(AFFECTS[hit]) or "（不触发）"))
    if all_flag:
        return sorted(SUITES), True, notes
    return got, False, notes


def changed_paths():
    """git 改动（含**已暂存**与**未跟踪** ✓ —— 只看 `git diff` 会漏新文件 ✗）。"""
    out = []
    for args in (["diff", "--name-only", "HEAD"],
                 ["ls-files", "--others", "--exclude-standard"]):
        try:
            r = subprocess.run(["git", "--no-pager", *args], cwd=str(ROOT),
                               capture_output=True, text=True, timeout=30)
        except Exception:                    # noqa: BLE001 —— 没有 git 就当空 ✓
            continue
        for ln in (r.stdout or "").splitlines():
            ln = ln.strip()
            if ln and ln not in out:
                out.append(ln)
    return out


def run_suites(names):
    """一个套件一个**子进程** ✓（Qt / torch 的全局状态互不影响 ✓）。"""
    bad = []
    for name in names:
        if name not in SUITES:
            print("  ⚠ 跳过未登记的套件 %s" % name)
            continue
        print("\n=== %s —— %s" % (name, SUITES[name]))
        r = subprocess.run([sys.executable, "-X", "utf8", "-m", "tools." + name],
                           cwd=str(ROOT))
        if r.returncode != 0:
            bad.append(name)
    print("\n%s" % ("全部通过 ✓" if not bad else "有失败的套件：%s ✗" % ", ".join(bad)))
    return 1 if bad else 0


def main(argv):
    args = [a for a in argv if not a.startswith("-")]
    flags = {a for a in argv if a.startswith("-")}
    if "--all" in flags:
        return run_suites(sorted(SUITES))
    paths = args or (changed_paths() if "--changed" in flags else [])
    if not paths:
        print("源 → 套件对照（改完先 `--changed` ✓ 只跑相关的 ✓）\n")
        for k in sorted(AFFECTS, key=lambda x: (x.count("/"), x)):
            print("  %-34s %s" % (k, ", ".join(AFFECTS[k]) or "（不触发）"))
        print("\n套件共 %d 个；`--changed` 按 git 改动算，`--run` 直接跑 ✓" % len(SUITES))
        return 0
    names, all_flag, notes = suites_for(paths)
    print("改动 %d 个文件 ⇒ %s\n" % (len(paths), "**必须全量**" if all_flag else "只跑这些 ✓"))
    for n in notes:
        print("  " + n)
    print("\n要跑的套件（%d）：%s" % (len(names), " ".join(names) or "（无 ✓）"))
    if "--run" in flags and names:
        return run_suites(names)
    if names:
        print("\n直接跑：python -m tools.selftest_index "
              + " ".join(paths) + " --run")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
