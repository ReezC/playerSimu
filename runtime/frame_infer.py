"""一帧的第 6 段：**一次推理 + 谁进决策**（阶段 1「编排收口」第二刀 ✓ 2026-10-20 ✓）。

搬自 `gui/live_thread.py::_run`（**行为零改动** ✓）：`model.predict` 那一处、
`split_dets` / `extra_class_dets` 的调用、"剪裁只喂视野框"的判定、掉落物单独留一份，
连注释一起搬过来 ✓（只把 `self.` 换成 `th.`、把 `_xxx` 局部名变成参数 ✓）。

⚠⚠ **主循环那次推理只许有一处** ✗（用户 2026-10-16 现场：我曾想在剪裁块里没认出玩家时
  "再跑一次整帧" ⇒ 当场被两条既有源码钉按住：`imgsz` 那条比较、弃用告警静音那条，
  都是靠"只有一处"才成立的 ✓）。要"退回整帧"只能用**上一拍记的账**
  （`th._vision_player_miss` ✓ 见 `_crop_on` 那段 ✓），不许加第二个推理口 ✗。

⚠ **偏移必须加回坐标，且全回路只许这一处**（`_det_off_xyxy` ✓）：模型看到的是剪裁块自己的
  坐标系，下游（追踪 / 决策 / 画框 / 掉落拾取）用的**全是全画面坐标** ✓ —— 漏了就是
  "所有框整体偏一块"、加两处就是"多偏一次"，两种都**不报错** ✓ 最难查 ✓。

⚠ **`split_dets` 是"谁进决策"的唯一出口**（怪 / 玩家 ✓）：掉落 / NPC / 其他玩家 / 宠物
  一律不进 ✗（`WorldState` 里根本没有它们的通道 ✓）。那几类只走 `extra_class_dets`
  ⇒ 给**画框**和**掉落拾取**用 ✓ 它**不喂任何 tracker** ✗。
  ⚠ 这两个函数**仍住在 `gui/live_thread.py`**（大量用例直接调 `lt.split_dets` ✓）⇒
  由调用方当参数传进来（`split=` / `extra=` ✓）—— 本模块因此**不 import gui** ✓。

判据 `selftest_live_panel::t_frame_infer_stage`：测谎中不推理（框自然不画 ✓）· 剪裁只在
"开着视野框检测**且**上一拍认出了玩家"时开 · 偏移只加一处 · 四路产物各归各位 ·
掉落单独一份且不进决策 · 推理耗时进窗口 ✓。
"""

import time

from core import perf
#: 掉落类（只给「自动拾取」算"接没接触"用 ✓ 见文件开头那条"不进决策"✓）
from perception.detect.classes import CLASS_DROP, CLASS_PET


def _det_crop_for(vis, vision_box, enabled):
    """⭐⭐ 「**只检测视野框**」（用户 2026-10-15 ✓）⇒ 返回 `(要喂给模型的图, 坐标偏移)`。

    · `enabled` 关 ⇒ **原图 + `(0, 0)`** ✓（老行为一字不变 —— 检测走整帧 ✓）；
    · 开 ⇒ 把 `vision_box` 那一块剪出来 + 偏移 `(left, top)` ✓。
      ⚠ 剪出来的是 **numpy 视图**（`vis[y0:y1, x0:x1]` 不拷内存 ✓ 每帧白拷 1 MB 没必要 ✗；
      `model.predict` 自己会做前处理 ✓）。
    · ⚠⚠ **调用方必须把偏移加回所有框坐标** ✗ —— 模型看到的是"剪裁块"自己的坐标系，
      而下游（追踪 / 决策 / 画框 / 掉落拾取）用的**全是全画面坐标** ✓。
    · ⚠ **退化框**（宽或高 ≤ 0）⇒ **退回原图** ✓：`vision_*` 是用户可填的，
      `left == right` 完全可能 ✓ —— 宁可多算一点，也**不许**把空图喂给模型 ✗
      （空数组会一路崩在 ultralytics 里，报错还看不出跟设置有关 ✗）。
    """
    if not enabled or vision_box is None:
        return vis, (0, 0)
    left, top, right, bottom = vision_box
    if right <= left or bottom <= top:
        return vis, (0, 0)
    return vis[top:bottom, left:right], (int(left), int(top))


def _det_off_xyxy(xyxy, det_off):
    """把剪裁推理的偏移**加回框坐标** ⇒ 交出去的永远是**全画面坐标** ✓。

    ⚠⚠ **全回路只许在这一处加** ✗（调用点唯一 ✓ 用例钉着 ✓）：模型给的是"剪裁块自己的
    坐标系"（原点在块左上角 ✓），而下游（`split_dets` / `extra_class_dets` / `_drop_boxes`
    / MobTracker / PlayerTracker / 画框）用的**全是全画面坐标** ✓ —— 加两处就**多偏一次** ✗、
    忘了加就**整体偏一块** ✗，两种都"能跑、不报错"✓ 最难查 ✓。

    `det_off == (0, 0)`（没剪裁 / 退回原图 ✓）⇒ **原样返回、不碰数组** ✓（省一次逐元素加法 ✓）。
    """
    if det_off == (0, 0):
        return xyxy
    return xyxy + (det_off[0], det_off[1], det_off[0], det_off[1])


class InferOut:
    """这一拍的推理产物（**本帧**数据 ✓ 交给后面的定位 / 决策 / 画框读 ✓）。

    · `mob` / `mob_low`        怪（高分 / 低分档 ✓ 低分档**只给 tracker 续命** ✓ 不进 `k`）
    · `player` / `player_low`  玩家候选（同上 ✓ 交给 `PlayerTracker` 挑"我"）
    · `cls_dets`               其余类别（掉落 / NPC / 其他玩家 / 宠物）⇒ **只给画框** ✓
    · `drop`                   掉落物框（只给「自动拾取」算"接没接触"✓）
    · `pets`                   ⭐ **宠物框**（只给**怪追踪的"遮挡"记账** ✓ 2026-10-22）——
                               用户现场："靠近怪物时，跟随玩家的宠物会把怪挡住导致检测框丢失"。
                               ⚠ 它**不是**一条"宠物进决策"的通道 ✗（`WorldState` 里没有、
                                 `ws.mobs` 里也不会多出东西 ✓）；只用来回答"这条怪轨迹这一拍
                                 是不是**被挡住了**、而不是**没了**" ✓（见 ByteTracker.update ✓）。
    · `k`                      这一帧检出几只怪（低分档不算 ✓ 状态行用）
    · `crop_off`               这一拍是整帧还是剪裁（`(0,0)` = 整帧 ✓ 打点 / 用例用）
    """

    __slots__ = ("mob", "mob_low", "player", "player_low", "cls_dets",
                 "drop", "pets", "k", "crop_off")

    def __init__(self, mob, mob_low, player, player_low, cls_dets, drop, k, crop_off,
                 pets=()):
        self.mob = mob
        self.mob_low = mob_low
        self.player = player
        self.player_low = player_low
        self.cls_dets = cls_dets
        self.drop = drop
        self.pets = tuple(pets or ())
        self.k = k
        self.crop_off = crop_off

    def __repr__(self):
        return "<InferOut mob=%d player=%d cls=%d k=%d>" % (
            len(self.mob), len(self.player), len(self.cls_dets), self.k)


def step(th, vis, model, det, _mob_class, _player_class, _extra_map,
         conf_mob, conf_player, _lie_mode, _vision_box, decision_settings, infer_ms,
         split_dets, extra_class_dets):
    """跑这一拍的那一次推理 ⇒ `InferOut`（测谎中 ⇒ 空产物，**战斗框自然不画** ✓）。

    :param det: 推理旋钮 `{conf, imgsz, device, iou, max_det, agnostic_nms, half_kw}`
                （`_run` 里都是**每秒热读**的局部量 ✓ 由调用方汇总成本帧这一份 ✓）
    :param _lie_mode: 测谎中 ⇒ **不推理**（四路产物全空 ✓ 但"显示那一帧 / 画 VT 描边"照旧 ✓）
    :param _vision_box: 这一拍的视野框 `(l, t, r, b)`（`None` = 没有 ✓ 见 `_det_crop_for`）
    :param infer_ms: 推理耗时窗口（最近 60 个 ✓ 状态行用 ✓ 直接往里 append ✓）
    :param split_dets: `gui/live_thread.split_dets`（**仍住在 gui** ✓ 由调用方传 ✓
                       见文件开头那条"不 import gui"✓）
    :param extra_class_dets: 同上 ✓
    :return: `InferOut`
    """
    # ---- 一次推理（整帧 / 或只视野框 ✓）：玩家（class 0）+ 怪（class 1）----
    t0 = time.perf_counter()
    if _lie_mode:
            # ⭐ 测谎中 ⇒ **不推理**（见上面那段 ✓）。下面那句
            #   `boxes = getattr(res, "boxes", None)` 对 `None` 是**防御式**的
            #   （`getattr(None, "boxes", None)` = None ✓）⇒ 这一拍 `mob_dets` /
            #   `player_cands` / `_cls_dets` 全空 ⇒ **战斗框自然不画** ✓，
            #   而"显示那一帧 / 画 VT 的描边"都照旧 ✓（画框段不受影响 ✓）。
            res = None
            _det_off = (0, 0)       # 没推理 ⇒ 无偏移 ✓（下面加回偏移要用它 ✓）
    else:
            # ⭐⭐ 「只检测视野框」（用户 2026-10-15 ✓ 见
            #   `DecisionSettings.vision_detect_only` ✓）：打开时只喂
            #   `vision_box` 那一块 ✓。**每帧现读**设置 ⇒ 面板上勾一下当场生效 ✓。
            #   ⚠ 它**不省推理时间** ✗：输入尺寸由 `imgsz` 定死（喂整帧 / 喂剪裁块，
            #     网络里跑的都是同一个 `imgsz` ✓）—— 换到的是**分辨率** ✓
            #     （见那个勾选框的 tooltip ✓）；想省时间得**同时**调小 `imgsz` ✓。
            #   ⚠ 偏移**必须**在下面加回 `xyxy`（只那一处 ✓），否则所有框整体位移 ✗。
            # ⭐⭐⭐ 2026-10-16 **用"上一拍记的账"决定这一拍剪不剪**（用户原话：
            #   "**现在不认玩家框了**"✓ + "**开启只检测视野框后，玩家检出框很容易
            #   永久定住不动**"✓）—— **剪裁块里没认出玩家 ⇒ 这一拍直接整帧检** ✓。
            #   · 病根同上（见上面那段 ⏳ 注释 ✓）：`vision_detect_only` 只把**视野框
            #     那一块**喂给模型 ✓，而**玩家也是 class 0** ✓ ⇒ 玩家一走出那一块
            #     ⇒ **一拍都检不到玩家** ✓；
            #   · ⚠⚠ **不加第二个推理口** ✗（主循环里那个口**只许一个** ✓
            #     两条既有源码钉钉着 ✓）⇒ 只能**拿上一拍的结论**当这一拍的依据 ✓
            #     （`_vision_player_miss` ✓ 在下面"玩家候选定稿"那一处写 ✓）；
            #   · 代价只在"接着几拍都没认出"的时候 ✓，**不是**每拍两次推理 ✗。
            _crop_on = bool(
                    getattr(decision_settings, "vision_detect_only", False)) \
                    and not getattr(th, "_vision_player_miss", False)
            _det_src, _det_off = _det_crop_for(
                    vis, _vision_box, _crop_on)
            # ⭐⭐ **漏检三旋钮接进推理**（用户 2026-10-16 ✓ 见上面那段说明 ✓）：
            #   · 去重门槛 = `iou_mob` ✓（默认 0.70 = 老行为一字不变 ✓；
            #     调大 ⇒ 挨得近的怪不被误去重 ⇒ 漏检少 ✓）；
            #   · 一帧上限 = `max_det` ✓（默认 300 = 框架默认 ✓；怪多时调 600 ✓）；
            #   · `agnostic_nms` = 类别互不压制 ✓（默认关 ✓ 也保持老行为 ✓）。
            #   ⚠ 三个都**热读** ⇒ 界面上改完当场生效 ✓（不用重开实时 ✓）。
            _kw = {"conf": det["conf"], "imgsz": det["imgsz"],
                          "device": det["device"], "iou": det["iou"],
                          "max_det": det["max_det"],
                          "agnostic_nms": det["agnostic_nms"]}
            _half_kw = det.get("half_kw") or {}
            res = model.predict(_det_src, verbose=False, **_kw, **_half_kw)[0]
            # ⏳ **① 待做（"视野框里没检出玩家 ⇒ 退回整帧"）的正确形状**（2026-10-16 ✓）：
            #   ⛔ 我第一版在下面**又加了一处推理调用** ✗ ⇒ 当场被两条既有源码钉按住：
            #      · 「主循环那次推理**不止一处了** ✗」
            #      · 「两处推理不是都走了那个**半精度开关**」 ✗（开关名别抄进来 ✗）
            #   ⇒ 这条路上**只许有一个推理口** ✓（`imgsz` 那条比较、弃用告警静音那条钉，
            #     都是靠"只有一处"才成立的 ✓）。
            #   ⭐ 正确形状 = **用"上一拍记的账"决定这一拍剪不剪** ✓（不加第二个推理口 ✓）：
            #     · 记一个 `th._vision_player_miss`（这一拍**剪裁块里有没有认出玩家** ✓
            #       在下面玩家候选定稿那处写 ✓）；
            #     · 下一拍：`_crop_on = 只检测视野框 and not 上一拍没认出玩家` ✓
            #       ⇒ 没认出就**这一拍直接整帧检** ✓（代价只在"接着没认出"的那几拍 ✓
            #       而不是"每拍两次推理" ✗）；
            #     · `det_crop_player_miss` 照打 ✓（让人一眼看出为什么退了整帧 ✓）。
            #   ⚠⚠ **写这条注释时我自己踩了仓库明文写过的那个坑** ✗：这两条钉是
            #     **按出现次数数源码文本**的 ✓ ⇒ 注释里原样抄一遍调用名就多一次、当场假红 ✓
            #     （`seq_editor.py` 里就留着同一条教训 ✓ 见开发日志 341/342 ✓）⇒ 本注释
            #     **不写那个调用名** ✓。
            # ⭐ 打个点，好让人在 `perf.log` 里**一眼看出**这一拍走的是哪条路 ✓
            #   （用户口径：改完要能自证生效 ✓；同 `crop_view_ok/miss` 那种写法 ✓）
            perf.count("det_crop" if _det_off != (0, 0) else "det_full")
            infer_ms.append((time.perf_counter() - t0) * 1000.0)
            perf.ms("infer_ms", t0)
            if len(infer_ms) > 60:
                    del infer_ms[0]

    #: ⚠ 这两个默认值原先在**块外**给（`_run` 里 "`_cls_dets = []`" 那一行 ✓）；
    #:   现在整段搬进来了 ⇒ 得在这里自己给 ✓（不然"没检出 / 测谎中"那一拍会 UnboundLocal ✗）
    _cls_dets = []
    k = 0
    mob_dets = []       # [(x1, y1, x2, y2, conf), ...] 给 MobTracker
    player_cands = []   # [(x1, y1, x2, y2, conf), ...] 交给 PlayerTracker
    #: ⭐ 两档低分档（2026-10-11 ✓）：只给 tracker 续命用 ✓ 这一拍没推理 ⇒ 空的 ✓
    mob_low = []
    player_low = []
    #: ⭐⭐ **掉落物框**（只给「自动拾取」算"接没接触"用 ✓ 用户 2026-10-06 ✓）——
    #:   原始检出 ✓ 不过 tracker ✓ 不进 `ws.mobs` ✓（见 `drop_touch` ✓）。
    #:   ⚠ 每帧都要先给默认（下面 `if boxes` 不成立时它就是空的 ✓）。
    _drop_boxes = []
    #: ⭐ **宠物框**（2026-10-22 ✓ 用户："靠近怪物时，跟随玩家的宠物会把怪挡住导致检测框丢失"）
    #:   —— 只当**遮挡物**喂给怪追踪（"这条轨迹是**被挡住**、不是**没了**"✓ 见
    #:   `ByteTracker.update(occluders=...)` ✓）；**不进** `ws.mobs` / `WorldState` ✓。
    _pet_boxes = []
    boxes = getattr(res, "boxes", None)
    if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            # ⭐⭐ 剪裁推理 ⇒ 把偏移**加回坐标**（**只在这一处** ✓ 用户 2026-10-15 ✓）：
            #   模型看到的是"剪裁块"自己的坐标系（原点在块左上角 ✓），而下游
            #   ——`split_dets` / `extra_class_dets` / `_drop_boxes`（自动拾取）/
            #   MobTracker / PlayerTracker / 画框——用的**全是全画面坐标** ✓
            #   ⇒ 就在"刚从 GPU 拿回来"这一步一次性挪回全画面坐标 ✓，
            #     后面谁都不用知道剪裁这件事 ✓。
            #   ⚠ 漏了这一步的形态是"**看着能跑、所有框整体偏一块**"✗（最难查的那种 ✓）。
            #   ⚠ **全回路只此一处** ✗（见 `_det_off_xyxy` 自己的说明 ✓）。
            xyxy = _det_off_xyxy(xyxy, _det_off)
            cfs = boxes.conf.cpu().numpy()
            try:
                    clss = boxes.cls.cpu().numpy().astype(int)
            except Exception:
                    clss = [1] * len(cfs)
            # ⭐⭐ **谁进决策**这条规矩在 `split_dets`（**唯一出口** ✓）
            #   —— 用户 2026-10-04 ✓ 要求："确认一下 Agent 认不认掉落、宠物类，
            #   或者说确认 Agent 没有将他们认成其他的"：
            #   **只有 `mob` / `player` 两类进得来** ✓，掉落 / NPC / 其他玩家 /
            #   宠物**一律不进** ✗ —— 而且 `WorldState` 里也只有 `mobs` 和
            #   `player` 两个字段 ⇒ 它们**根本没有通道** ✓（用例拿 6 个类别
            #   当场跑过 ✓ 见 `selftest_decision.t_drop_pet_never_enter_decision`）。
            # ⚠ 返回值是**三个**（2026-10-11 起 ✓ 多了 `mob_low` 低分档 ✓
            #   见 `split_dets` 那段说明 ✓）—— 它**只喂 tracker 续命** ✓
            #   **不进 `k`**（`k` 是"这一帧检出几只怪"的计数 ✓ 低分档不算 ✓）。
            # ⚠ 返回值是**四个**（2026-10-11 起 ✓ 两档低分档：怪 `mob_low` /
            #   玩家 `player_low` ✓ 见 `split_dets` ✓）—— 它们**只喂 tracker 续命** ✓
            #   **不进 `k`**（`k` 是"这一帧检出几只怪"✓ 低分档不算 ✓）。
            mob_dets, player_cands, mob_low, player_low = split_dets(
                xyxy, cfs, clss, _mob_class, _player_class,
                conf_mob, conf_player)
            k = len(mob_dets)      # = 原来的 `k += 1`（怪框计数 ✓ 等价 ✓）
            # ⭐⭐ **记下"这一拍剪裁块里到底有没有认出玩家"** —— 供**下一拍**决定
            #   剪不剪（用户 2026-10-16 ✓ 见上面 `_crop_on` 那段 ✓）。
            #   ⚠ 只认"**这一帧真检出**"（`player_cands` 非空 ✓）—— **不看
            #     tracker 最后有没有框** ✗：那份会被"低分维护"顶住（框定住不动 ✓
            #     就是那个毛病的另一半 ✓）⇒ 也算成"认出来了" ⇒ 视野框里其实没人
            #     却不退回整帧 ✗。
            th._vision_player_miss = not player_cands
            # ⭐⭐ **同一批原始框里另外那几类**（掉落 / NPC / 其他玩家 / 宠物 ✓
            #   用户 2026-10-06 ✓"为何没看到掉落物、宠物的检出框？"）—— 给
            #   **绘制层**（`_draw_show_frame` ✓ 每一类一个勾，见设置→界面→
            #   检测框颜色 ✓）。
            #   ⛔ 它**不喂任何 tracker** ✗：`ws.mobs` 仍旧只由 `mob_dets` 喂 ✓
            #     （"谁进决策"只有 `split_dets` 那一个出口 ✓ 用例钉着 ✓）。
            #   ⚠ `clss` 读不出来时那一段走的是 `clss = [1] * len(cfs)`（歪打正着：
            #     全当怪 ✓）⇒ 这里自然也只会挑出"怪"、不会凭空多出别的类 ✓。
            _cls_dets = extra_class_dets(xyxy, cfs, clss, _extra_map)
            # ⭐⭐ **掉落物那一类单独留一份**（用户 2026-10-06 ✓ 自动拾取）——
            #   ⚠ 只挑**掉落**（`CLASS_DROP` ✓）：宠物/NPC/其他玩家**不参与**
            #     拾取判定（它们本来就只是"画出来给人看"✓ 见上面那批 ✓）。
            #   ⚠ 用 `extra_map`（模型类号 → 我们的类号 ✓）判 = **按名字对齐**过
            #     的那套 ✓（权重换过类的号也不会张冠李戴 ✓）。
            _drop_boxes = [(float(_x1), float(_y1), float(_x2), float(_y2))
                                          for (_x1, _y1, _x2, _y2), _c, _cls
                                          in zip(xyxy, cfs, clss)
                                          if _extra_map.get(int(_cls)) == CLASS_DROP]
            # ⭐⭐ **宠物框另留一份当"遮挡物"**（2026-10-22 ✓ 用户现场：
            #   "靠近怪物时，跟随玩家的宠物会把怪挡住导致检测框丢失"）。
            #   宠物贴着角色走 ⇒ 近身那只怪常常正好在宠物后面 ⇒ 模型**这一拍看不见它** ✗
            #   ⇒ 老的追踪只能按"漏检"处理：`missed` 往上涨、涨过「失联帧长」就换 id / 丢目标
            #   ⇒ 决策层看到的是"怪没了" ⇒ 打到一半掉头 / 反复锁新目标 ✓。
            #   ⚠ 只当**遮挡物**用：`ws.mobs` 仍旧只由 `mob_dets` 喂 ✓（宠物本身**永远不进
            #     `mobs` / `WorldState`** ✓ 那条红线不动 ✓ 用例 `t_drop_pet_never_enter_decision` ✓）。
            _pet_boxes = [(float(_x1), float(_y1), float(_x2), float(_y2))
                                      for (_x1, _y1, _x2, _y2), _c, _cls
                                      in zip(xyxy, cfs, clss)
                                      if _extra_map.get(int(_cls)) == CLASS_PET]

    return InferOut(mob=mob_dets, mob_low=mob_low, player=player_cands,
                    player_low=player_low, cls_dets=_cls_dets,
                    drop=_drop_boxes, k=k, crop_off=_det_off,
                    pets=_pet_boxes)
