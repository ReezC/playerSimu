"""被控机（A 机）部署台。

    python -m deploy.app          （或双击仓库根的「被控机部署台.bat」）

把 docs/A_SETUP.md 里那几条手工命令做成按钮：时钟对时 / 时间码探针 /
键盘中继 / 屏幕推流 / 小地图推流。参数改一下立刻写进 config/deploy.json，
下次打开原样复现。

**A 机上不敲命令**：连「小地图要框哪一块」也是这里的按钮（「小地图推流」卡片
→ 框选…）。框选走 tools/minimap_push.ask_region —— 和命令行 --pick 是同一份
实现，框之前会把本窗口**先藏起来**再抓屏，不然截到的图里盖着部署台自己。

**版面为什么这么排**
    左边一列 5 张服务卡片，顺序 = 部署顺序（对时 → 探针 → 键盘 → 推流 → 小地图）；
    右上角是环境自检（「按了没反应」的九成原因都在那几条里），
    右下角是几个服务混在一起的运行日志（可筛选）。
    改参数、启停、看日志、看自检全在一屏里 —— 不用开第二个窗口，也不用回头翻手册。

**每张卡片底部都写着「这一项实际会执行的命令行」**（可选中复制）
    界面上做的任何事都等价于手敲那条命令：对不上时一眼就看得出，
    想手工排查时直接复制出去跑。
"""

from collections import deque

import faulthandler
import html
import sys
import threading
import time
import traceback
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# 这一段必须排在**依赖导入之前**：依赖没装时（A 机上最常见 —— 没跑过
# pip install -r deploy/requirements.txt）模块级的 import PyQt5 会直接抛异常，
# 而 pythonw 没有控制台，那份 traceback 打给 None 就没了，末尾
# `if __name__ == "__main__"` 那段根本到不了 —— 不弹框、不写日志，
# 表现还是「双击没反应」。所以错误上报先定义好，并包住下面那些导入。
# ─────────────────────────────────────────────────────────────────────
from deploy import config as dcfg

APP_NAME = "playerSimu 部署台"
CRASH_LOG = dcfg.ROOT / "deploy_crash.log"

# 已经报过一次错了（写日志 + 弹框）。异常往外抛之后还会过 sys.excepthook，
# 没有这个标记那份堆栈会被写两遍。
_reported = False


def _report_startup_error():
    """出错就留证据 + 弹框，**绝不静默退出**。

    **为什么非要弹框**：这个界面是用 pythonw 起的（为了不留黑窗），一旦在窗口
    出现之前就挂了，用户看到的就是「双击没反应」—— 排查时间全耗在「不知道哪错了」
    上（依赖没装、sys.stderr 是 None 的老坑、Qt 起不来……都归它管）。

    用 ctypes 弹而不是 QMessageBox：崩在 QApplication 起来之前时 Qt 还不能用。
    这里自己也不许再抛，否则就白写了。
    """
    global _reported
    tb = traceback.format_exc()
    try:
        with open(CRASH_LOG, "a", encoding="utf-8") as fh:
            fh.write("\n=== 启动失败 %s ===\n%s\n"
                     % (time.strftime("%Y-%m-%d %H:%M:%S"), tb))
    except Exception:
        pass
    _reported = True
    if sys.stderr is not None:
        sys.stderr.write(tb)

    head = tb.strip().splitlines()[-1] if tb.strip() else "未知错误"
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            None,
            "部署台启动失败：\n\n%s\n\n完整堆栈已写进：\n%s" % (head, CRASH_LOG),
            APP_NAME, 0x10)          # MB_ICONERROR
    except Exception:
        pass


try:
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal
    from PyQt5.QtGui import QFont
    from PyQt5.QtWidgets import (QAction, QApplication, QCheckBox, QFileDialog,
                                 QFrame, QGridLayout, QHBoxLayout, QLabel,
                                 QLineEdit, QMainWindow, QMessageBox,
                                 QPlainTextEdit, QPushButton, QScrollArea,
                                 QSizePolicy, QSplitter, QToolBar, QVBoxLayout,
                                 QWidget)

    from deploy import selfcheck, services
    from tools import push_presets as pp
    from deploy.runner import Proc
    from deploy.settings_dialog import DeploySettingsDialog
    from gui.widgets import (NoWheelComboBox, NoWheelDoubleSpinBox,
                             NoWheelSpinBox)
except Exception:
    # 最常见的就是 PyQt5 / PyYAML / pyserial 没装（见 deploy/requirements.txt）——
    # 让上面那个框把话说清楚，别让人对着一个不会出现的窗口发呆。
    _report_startup_error()
    raise

# 每个服务一色：日志行、状态点、卡片左边框都用它。选的是深色底上看得清的中等饱和度。
SERVICE_COLOR = {
    "clock": "#1a73e8",     # 蓝
    "probe": "#188038",     # 绿
    "kbd": "#b06000",       # 橙
    "push": "#7b1fa2",      # 紫
    "mmap": "#00796b",      # 青
}
UI_COLOR = "#5f6368"        # 界面自身的提示（不属于任何一个服务）

LEVEL_COLOR = {"ok": "#188038", "warn": "#b06000", "bad": "#d93025"}

QSS = """
QMainWindow, QWidget#Central { background: #f0f2f5; }
QWidget { color: #202124; }

QFrame#Card {
    background: #ffffff;
    border: 1px solid #e2e5ea;
    border-radius: 8px;
}
QLabel#CardTitle { font-weight: 600; }
QLabel#CardSub { color: #80868b; }
QLabel#Mono {
    font-family: Consolas, "Courier New", monospace;
    color: #3c4043; background: #f8f9fa;
    border: 1px solid #e2e5ea; border-radius: 4px; padding: 4px 6px;
}
QLabel#SelfRow { color: #202124; }
QLabel#SelfDetail { color: #80868b; }
QLabel#Chip { color: #5f6368; }
QLabel#HeadTitle { font-weight: 600; }

QPlainTextEdit#Log {
    background: #202124; color: #e8eaed;
    border: 1px solid #dadce0; border-radius: 6px;
}

QPushButton {
    background: #ffffff; border: 1px solid #dadce0; border-radius: 5px;
    padding: 4px 10px; color: #202124;
}
QPushButton:hover { background: #f8f9fa; border-color: #c9cdd4; }
QPushButton:pressed { background: #f1f3f4; }
QPushButton:disabled { background: #f8f9fa; color: #9aa0a6; border-color: #e2e5ea; }
QPushButton#Primary {
    background: #1a73e8; color: #ffffff; border: none; font-weight: 600;
}
QPushButton#Primary:hover { background: #4285f4; }
QPushButton#Primary:disabled { background: #dadce0; color: #ffffff; }
QPushButton#Danger { color: #d93025; }

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #ffffff; border: 1px solid #dadce0; border-radius: 5px;
    padding: 3px 7px; min-height: 20px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #4285f4;
}
QComboBox::drop-down { border: none; width: 18px; }
QComboBox QAbstractItemView {
    background: #ffffff; border: 1px solid #dadce0;
    selection-background-color: #e8f0fe; selection-color: #202124;
}

QScrollArea { background: transparent; border: none; }
QSplitter::handle { background: #e2e5ea; }
QSplitter::handle:vertical { height: 6px; }
"""


# ══════════════════════════════════════════════════════════════
# 服务卡片
# ══════════════════════════════════════════════════════════════
class ServiceCard(QFrame):
    """一个服务：状态 + 参数 + 启动/停止 + 将执行的命令行。

    参数控件按 services.PARAMS 的参数表生成 —— 界面里能改的东西和命令里能传的
    东西是同一份定义，不会对不上。
    """

    changed = pyqtSignal()
    start_clicked = pyqtSignal(str)
    stop_clicked = pyqtSignal(str)

    def __init__(self, key, cfg, parent=None):
        super().__init__(parent)
        self.key = key
        self.cfg = cfg
        self.setObjectName("Card")
        self._getters = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        root.addLayout(self._build_head())
        form = QGridLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)
        root.addLayout(form)
        for row, spec in enumerate(services.PARAMS[key]):
            self._add_param(form, row, spec, cfg)
        root.addLayout(self._build_cmd_row())
        self.setMinimumWidth(520)

    # ---------------- 头部 ----------------

    def _build_head(self):
        head = QHBoxLayout()
        head.setSpacing(8)

        self.lamp = QLabel("●")
        self.lamp.setStyleSheet("color: %s;" % SERVICE_COLOR[self.key])
        head.addWidget(self.lamp)

        title = QLabel(services.TITLE[self.key])
        title.setObjectName("CardTitle")
        head.addWidget(title)

        sub = QLabel(services.SUB[self.key])
        sub.setObjectName("CardSub")
        head.addWidget(sub)
        head.addStretch(1)

        self.lbl_state = QLabel("未运行")
        self.lbl_state.setObjectName("CardSub")
        head.addWidget(self.lbl_state)

        self.btn_start = QPushButton("启动")
        self.btn_start.setObjectName("Primary")
        self.btn_start.clicked.connect(lambda: self.start_clicked.emit(self.key))
        head.addWidget(self.btn_start)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.setObjectName("Danger")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(lambda: self.stop_clicked.emit(self.key))
        head.addWidget(self.btn_stop)
        return head

    # ---------------- 参数 ----------------

    def _add_param(self, form, row, spec, cfg):
        label = QLabel(spec["label"])
        label.setToolTip(spec.get("tip", ""))
        form.addWidget(label, row, 0)

        widget, getter = self._make_widget(spec, cfg)
        widget.setToolTip(spec.get("tip", ""))
        if spec.get("width"):
            widget.setMinimumWidth(spec["width"])
        holder = widget
        if spec["kind"] == "path":
            row_box = QHBoxLayout()
            row_box.setSpacing(4)
            row_box.addWidget(widget, 1)
            btn = QPushButton("浏览…")
            btn.setFixedWidth(58)
            btn.clicked.connect(lambda _c, w=widget, s=spec: self._pick_file(w, s))
            row_box.addWidget(btn)
            holder = QWidget()
            holder.setLayout(row_box)
        elif spec["kind"] == "serial":
            row_box = QHBoxLayout()
            row_box.setSpacing(4)
            row_box.addWidget(widget, 1)
            btn = QPushButton("刷新")
            btn.setFixedWidth(48)
            btn.setToolTip("重新列出当前插着的串口")
            btn.clicked.connect(lambda _c, w=widget: self._fill_ports(w))
            row_box.addWidget(btn)
            holder = QWidget()
            holder.setLayout(row_box)
        elif spec["kind"] == "region":
            row_box = QHBoxLayout()
            row_box.setSpacing(4)
            row_box.addWidget(widget, 1)
            btn = QPushButton("框选…")
            btn.setFixedWidth(58)
            btn.setToolTip("抓一张全屏图，把游戏的小地图面板框出来\n"
                           "（框之前这个窗口会自动藏起来）")
            btn.clicked.connect(lambda _c, w=widget: self._pick_region(w))
            row_box.addWidget(btn)
            holder = QWidget()
            holder.setLayout(row_box)

        form.addWidget(holder, row, 1)
        form.setColumnStretch(1, 1)
        # (参数表, 控件, 取值函数)：控件也存下来 —— 只有它能把值写回去
        self._getters[spec["key"]] = (spec, widget, getter)

    def _make_widget(self, spec, cfg):
        """按参数表建控件。返回 (控件, 取值函数)。"""
        kind = spec["kind"]
        val = cfg.get(spec["key"])

        if kind == "int":
            w = NoWheelSpinBox()
            w.setRange(int(spec.get("minimum", 0)), int(spec.get("maximum", 10 ** 9)))
            try:
                w.setValue(int(val))
            except (TypeError, ValueError):
                w.setValue(int(spec.get("minimum", 0)))
            w.valueChanged.connect(self.changed)
            return w, w.value

        if kind == "float":
            w = NoWheelDoubleSpinBox()
            w.setDecimals(int(spec.get("decimals", 3)))
            w.setRange(float(spec.get("minimum", -1e9)), float(spec.get("maximum", 1e9)))
            w.setSingleStep(float(spec.get("step", 0.01)))
            try:
                w.setValue(float(val))
            except (TypeError, ValueError):
                w.setValue(0.0)
            w.valueChanged.connect(self.changed)
            return w, w.value

        if kind == "check":
            w = QCheckBox()
            w.setChecked(bool(val))
            w.toggled.connect(self.changed)
            return w, w.isChecked

        if kind == "choice":
            w = NoWheelComboBox()
            for text, data in spec.get("choices", ()):
                w.addItem(text, data)
            i = w.findData(val)
            w.setCurrentIndex(i if i >= 0 else 0)
            w.currentIndexChanged.connect(self.changed)
            return w, w.currentData

        if kind == "combo_edit":
            w = NoWheelComboBox()
            w.setEditable(True)
            w.addItems([str(c) for c in spec.get("choices", ())])
            w.setCurrentText(str(val if val is not None else ""))
            w.currentTextChanged.connect(self.changed)
            cast = spec.get("cast", str)
            return w, (lambda: _cast(w.currentText(), cast, val))

        if kind == "size":
            w = NoWheelComboBox()
            w.setEditable(True)
            w.addItems([str(c) for c in spec.get("choices", ())])
            try:
                cur = "%dx%d" % (int(cfg.get(spec["keys"][0])),
                                 int(cfg.get(spec["keys"][1])))
            except (TypeError, ValueError):
                cur = str(spec.get("choices", ("1366x768",))[0])
            w.setCurrentText(cur)
            w.currentTextChanged.connect(self.changed)
            return w, w.currentText

        if kind == "region":
            w = QLineEdit(_region_text(cfg))
            w.setPlaceholderText("x,y,w,h —— 点右边「框选…」")
            w.textChanged.connect(self.changed)
            return w, (lambda: w.text().strip())

        if kind == "serial":
            w = NoWheelComboBox()
            w.setEditable(True)
            w.setMinimumWidth(spec.get("width", 150))
            w.setCurrentText(str(val or ""))
            self._fill_ports(w, keep=str(val or ""))
            w.currentTextChanged.connect(self.changed)
            return w, (lambda: w.currentText().strip())

        # text / path 都是普通单行输入
        w = QLineEdit(str(val if val is not None else ""))
        if kind == "path":
            w.setPlaceholderText("留空 = 用 PATH 里的命令" if
                                 spec["key"] == "ffmpeg" else "")
        w.textChanged.connect(self.changed)
        return w, (lambda: w.text().strip())

    @staticmethod
    def _fill_ports(combo, keep=None):
        """把当前插着的串口填进下拉（可编辑，不在列表里的也能手输）。"""
        keep = combo.currentText() if keep is None else keep
        try:
            from serial.tools import list_ports
            ports = [p.device for p in list_ports.comports()]
        except Exception:
            ports = []
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(ports)
        combo.setCurrentText(keep)
        combo.blockSignals(False)

    def _pick_file(self, line, spec):
        path, _f = QFileDialog.getOpenFileName(
            self, "选择文件", line.text() or str(dcfg.ROOT),
            spec.get("filt", "所有文件 (*)"))
        if path:
            # 能转成相对仓库根的路径就转 —— 配置换机器/换目录后还能用
            try:
                path = str(Path(path).relative_to(dcfg.ROOT))
            except ValueError:
                pass
            line.setText(path)

    def _pick_region(self, line):
        """「框选…」：抓屏拖一个框 → 把 x,y,w,h 写回输入框。

        走 `tools.minimap_push.ask_region`（**和命令行 --pick 同一份实现**）：
        owner 传自己，它会先把部署台窗口藏起来再抓屏 —— 不藏的话截到的图里
        盖着部署台自己，人对着自己的界面框小地图（这一步不做就白框）。

        写回输入框靠 `setText` 触发 textChanged → 部署台照常「立即落盘 + 刷新
        命令行预览」，不另走一条路 —— 否则「框完没保存」这种坑迟早会出现。
        """
        try:
            from tools.minimap_push import ask_region
        except Exception as e:
            QMessageBox.warning(self, "框选不可用",
                                "加载框选模块失败：%s: %s" % (type(e).__name__, e))
            return
        try:
            rect = ask_region(owner=self)
        except Exception as e:
            # 抓屏失败（没有可用屏幕 / 远程会话）也要说清楚，别静默什么都不发生
            QMessageBox.warning(self, "框选失败", "%s: %s" % (type(e).__name__, e))
            return
        if not rect:
            return
        line.setText(",".join(str(int(v)) for v in rect))

    # ---------------- 命令行预览 ----------------

    def _build_cmd_row(self):
        row = QHBoxLayout()
        row.setSpacing(6)
        tag = QLabel("将执行")
        tag.setObjectName("CardSub")
        row.addWidget(tag)

        self.lbl_cmd = QLabel("")
        self.lbl_cmd.setObjectName("Mono")
        self.lbl_cmd.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.lbl_cmd.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        row.addWidget(self.lbl_cmd, 1)

        btn = QPushButton("复制")
        btn.setFixedWidth(52)
        btn.setToolTip("复制这条命令（可以直接贴到 PowerShell 里手动跑）")
        btn.clicked.connect(self._copy_cmd)
        row.addWidget(btn)
        return row

    def _copy_cmd(self):
        QApplication.clipboard().setText(self.cmd_text())
        self.lbl_cmd.setToolTip("已复制到剪贴板")

    def cmd_text(self):
        return services.format_cmd(services.build_cmd(self.key, self.values()))

    def refresh_cmd(self):
        text = self.cmd_text()
        self.lbl_cmd.setText(text)
        self.lbl_cmd.setToolTip(text)      # 长命令被省略号截断时，悬停看全文

    # ---------------- 取值 / 状态 ----------------

    def values(self):
        """参数控件的当前值 → 配置片段（按 spec 的 keys 摊平）。"""
        out = {}
        for spec, _w, getter in self._getters.values():
            val = getter()
            if spec["kind"] == "size":
                w, h = _parse_size(val)
                out[spec["keys"][0]] = w
                out[spec["keys"][1]] = h
            elif spec["kind"] == "region":
                # 一个框 → 四个键。解析不出来时写 None（**不猜一个默认区域**）：
                # 猜出来的区域会推一片无关画面出去，B 机那边「底图对不上」，
                # 看着完全像寻路坏了；写 None 则会被 missing_hint 当场拦住。
                for k, v in zip(spec["keys"], _parse_region(val)):
                    out[k] = v
            else:
                out[spec["keys"][0]] = val
        return out

    def set_running(self, running, uptime=0.0):
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        if running:
            self.lbl_state.setText("运行中 %s" % _hms(uptime) if uptime
                                   else "运行中")
            self.lamp.setStyleSheet("color: %s;" % SERVICE_COLOR[self.key])
        else:
            self.lbl_state.setText("未运行")
            self.lamp.setStyleSheet("color: #dadce0;")


def _cast(text, cast, fallback):
    try:
        return cast(str(text).strip())
    except (TypeError, ValueError):
        return fallback


def _parse_size(text):
    """'1366x768' → (1366, 768)；解析不出来退回 1366x768。"""
    try:
        a, b = str(text).lower().replace("×", "x").split("x")
        return int(float(a)), int(float(b))
    except Exception:
        return 1366, 768


def _region_text(cfg):
    """配置里的 x/y/w/h → 输入框显示文本 'x,y,w,h'；没框选过就是空串。"""
    vals = [(cfg or {}).get(k) for k in ("x", "y", "w", "h")]
    if any(v in (None, "") for v in vals):
        return ""
    try:
        return ",".join(str(int(v)) for v in vals)
    except (TypeError, ValueError):
        return ""


def _parse_region(text):
    """'x,y,w,h' → (x, y, w, h)；解析不出来返回 (None, None, None, None)。

    逗号/中文逗号/空格都认（手输时全角逗号很常见）。**不退回默认值** ——
    理由见 ServiceCard.values()。
    """
    parts = [v for v in
             str(text or "").replace("，", ",").replace(" ", ",").split(",")
             if v != ""]
    if len(parts) != 4:
        return (None, None, None, None)
    try:
        return tuple(int(float(v)) for v in parts)
    except (TypeError, ValueError):
        return (None, None, None, None)


def _hms(sec):
    sec = int(max(0, sec))
    if sec >= 3600:
        return "%d:%02d:%02d" % (sec // 3600, sec % 3600 // 60, sec % 60)
    return "%d:%02d" % (sec // 60, sec % 60)


# ══════════════════════════════════════════════════════════════
# 环境自检
# ══════════════════════════════════════════════════════════════
class SelfCheckPane(QWidget):
    """右上角：ffmpeg / 串口 / B 机 / 证书 / 与 link.yaml 是否一致。"""

    rerun = pyqtSignal()
    apply_link = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        head = QHBoxLayout()
        head.setSpacing(6)
        title = QLabel("环境自检")
        title.setObjectName("CardTitle")
        head.addWidget(title)
        self.lbl_when = QLabel("检查中…")
        self.lbl_when.setObjectName("CardSub")
        head.addWidget(self.lbl_when)
        head.addStretch(1)

        self.btn_apply = QPushButton("按 link.yaml 填")
        self.btn_apply.setToolTip(
            "把 config/link.yaml 里的 B 机 IP、端口、分辨率、串口号填到部署台。\n"
            "两边不一致时的现象是「流发出去了，B 机就是没画面」。")
        self.btn_apply.clicked.connect(self.apply_link)
        head.addWidget(self.btn_apply)

        btn = QPushButton("重新自检")
        btn.clicked.connect(self.rerun)
        head.addWidget(btn)
        root.addLayout(head)

        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(2)
        box = QWidget()
        box.setObjectName("Card")
        inner = QVBoxLayout(box)
        inner.setContentsMargins(10, 8, 10, 8)
        inner.setSpacing(4)
        inner.addLayout(self.body)
        self.rows = []
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(box)
        root.addWidget(scroll, 1)
        self._box = box

    def set_items(self, items):
        for lbl in self.rows:
            lbl.setParent(None)
        self.rows = []
        for it in items:
            color = LEVEL_COLOR.get(it["level"], "#5f6368")
            text = '<span style="color:%s">●</span> %s' % (
                color, html.escape(it["title"]))
            if it.get("detail"):
                text += '<br><span style="color:#80868b; font-size:11px">%s</span>' % (
                    html.escape(it["detail"]).replace("\n", "<br>"))
            lbl = QLabel(text)
            lbl.setObjectName("SelfRow")
            lbl.setWordWrap(True)
            lbl.setTextFormat(Qt.RichText)
            self.body.addWidget(lbl)
            self.rows.append(lbl)
        self.lbl_when.setText("已于 %s 检查" % time.strftime("%H:%M:%S"))


# ══════════════════════════════════════════════════════════════
# 日志
# ══════════════════════════════════════════════════════════════
class LogPane(QWidget):
    """右下角：四个服务的输出混在一起，按服务可筛。"""

    ERROR_WORDS = ("error", "traceback", "failed", "失败", "错误", "拒绝", "无法")

    #: 保留多少行历史（和 txt 的 maximumBlockCount 一致）。筛选切换要能重排，
    #: 所以必须自己留一份缓冲 —— 只靠控件里的内容做不到「切回来还能看见」。
    #: 默认值与 deploy/config.py 的 log.max_lines 一致（设置弹窗里能改）
    HISTORY = 4000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.filter = "all"
        # (时间戳/占位, 服务 key, 该行文本, 颜色)。筛选切换时按它重排。
        self._buf = deque(maxlen=self.HISTORY)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        head = QHBoxLayout()
        head.setSpacing(6)
        title = QLabel("运行日志")
        title.setObjectName("CardTitle")
        head.addWidget(title)

        self.cmb = NoWheelComboBox()
        self.cmb.addItem("全部", "all")
        for key in services.ORDER:
            self.cmb.addItem(services.TITLE[key], key)
        self.cmb.setToolTip("只影响显示，不影响正在跑的服务")
        self.cmb.currentIndexChanged.connect(self._on_filter)
        head.addWidget(self.cmb)
        head.addStretch(1)

        for text, slot, tip in (("清空", self.clear, "清掉已显示的内容"),
                                ("保存…", self.save, "把当前显示的内容存成 txt")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            head.addWidget(b)
        root.addLayout(head)

        self.txt = QPlainTextEdit()
        self.txt.setObjectName("Log")
        self.txt.setReadOnly(True)
        self.txt.setMaximumBlockCount(self.HISTORY)
        self.txt.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        f = QFont("Consolas")
        f.setStyleHint(QFont.Monospace)
        self.txt.setFont(f)
        root.addWidget(self.txt, 1)

    def set_history(self, n):
        """改「保留多少行历史」（设置弹窗里改的就是它）。

        **两处必须一起改**（`_buf` 的历史缓冲 + 控件的 maximumBlockCount）：只改一边
        就会出现"切一下筛选能看见、控件里却没有"这种对不上的怪象。
        `deque` 的 maxlen 建好就改不了 —— 所以连同内容重建一个，旧行按新上限保留
        （调小会立刻丢掉较旧的行，这也正是界面上 tooltip 说清楚的代价）。
        """
        n = max(100, int(n))
        self.HISTORY = n
        self._buf = deque(self._buf, maxlen=n)
        self.txt.setMaximumBlockCount(n)

    def _on_filter(self, _i):
        self.filter = self.cmb.currentData()
        self._rerender()

    def _visible(self, key):
        """这一条在当前筛选下该不该显示。

        「界面消息」（ui）永远显示 —— 它说的是启停/自检结果，筛掉会让人以为
        界面没反应。
        """
        return self.filter == "all" or key in ("ui", self.filter)

    def append(self, key, text):
        color = SERVICE_COLOR.get(key, UI_COLOR)
        low = str(text).lower()
        if any(w in low for w in self.ERROR_WORDS):
            color = "#f28b82"
        stamp = time.strftime("%H:%M:%S")
        pad = "&nbsp;" * len(stamp)
        lines = str(text).splitlines() or [""]
        show = self._visible(key)
        for i, line in enumerate(lines):
            item = (stamp if i == 0 else pad, key, line, color)
            self._buf.append(item)          # 历史都留着（有上限）
            if show:
                self._write(*item)
        if show:
            sb = self.txt.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _write(self, prefix, key, line, color):
        """把一条写进控件（与筛选无关，重排时也用它）。"""
        tag = "" if key == "ui" else (
            '<span style="color:%s">[%s]</span> '
            % (SERVICE_COLOR.get(key, UI_COLOR), html.escape(services.TITLE[key][:2])))
        self.txt.appendHtml(
            '<span style="color:#9aa0a6">%s</span> %s'
            '<span style="color:%s">%s</span>'
            % (prefix, tag, color, html.escape(line)))

    def _rerender(self):
        """按当前筛选把历史重排一遍。

        **必须留一份缓冲**：原来筛选只判断「新来的行」，切到别的服务时已经显示的
        内容不动、被挡下的历史也补不回来 —— 表现是「换了筛选看着没生效」，
        而且**切到 A 服务时 B 服务的历史被永久丢掉**（切过去只剩空白），
        排查问题时最需要的那段日志正好没了。
        """
        self.txt.setUpdatesEnabled(False)
        try:
            self.txt.clear()
            for item in self._buf:
                if self._visible(item[1]):
                    self._write(*item)
        finally:
            self.txt.setUpdatesEnabled(True)
        sb = self.txt.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear(self):
        self._buf.clear()
        self.txt.clear()

    def save(self):
        path, _f = QFileDialog.getSaveFileName(
            self, "保存日志", str(dcfg.ROOT / "deploy_log.txt"),
            "文本文件 (*.txt)")
        if not path:
            return
        try:
            Path(path).write_text(self.txt.toPlainText(), encoding="utf-8")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))


# ══════════════════════════════════════════════════════════════
# 主窗口
# ══════════════════════════════════════════════════════════════
class DeployWindow(QMainWindow):
    sig_line = pyqtSignal(str, str)      # 服务 key, 一行日志（从读取线程发过来）
    sig_exit = pyqtSignal(str, object)   # 服务 key, 退出码
    sig_check = pyqtSignal(list)         # 自检结果
    sig_sweep_done = pyqtSignal(object, object)   # 推流自检跑完（rows, 落盘路径）

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.cfg = dcfg.read()
        self.procs = {}
        self.cards = {}
        self._stopping = set()
        self._hinted = set()        # 已跟过的「错误翻译」，同一条只跟一次

        self.sig_line.connect(self._on_line)
        self.sig_exit.connect(self._on_exit)
        self.sig_check.connect(self._on_check)
        self.sig_sweep_done.connect(self._on_sweep_done)
        self._sweep = None            # 推流自检线程（跑的时候非空）

        self._build()
        self._build_toolbar()
        self.setStyleSheet(QSS)
        self.resize(int(self.cfg["window"]["w"]), int(self.cfg["window"]["h"]))

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(300)
        self._save_timer.timeout.connect(self._save_now)

        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._refresh_states)
        self._tick.start()

        self._sync_cards_from_cfg()
        self.log("ui", "设置自动保存在 %s" % (dcfg.PATH.name,))
        self.log("ui", "部署手册：docs/A_SETUP.md")
        self.run_selfcheck()

    # ---------------- 界面 ----------------

    def _build_toolbar(self):
        """工具栏：目前只有「推流自检」，以后要放别的整机级动作也放这里。

        为什么值得有这条：**A 机扛不扛得住、哪套参数延迟最低**，原先只能靠人一条条
        手改参数、手记数字。自检把它变成"各点一次、等几分钟、拿一张表"，
        而延迟那一半由 B 机量（两边按同一份清单顺序走，墙钟对账，见 tools/push_presets.py）。
        """
        bar = QToolBar("工具")
        bar.setObjectName("Tools")
        bar.setMovable(False)
        self.addToolBar(bar)

        act = QAction("推流自检…", self)
        act.setToolTip(
            "按 config/push_presets.json 的候选清单**逐条换参试推**，采 A 机侧指标\n"
            "（ffmpeg 的 speed / 实际帧率 / 丢帧 / 码率 / CPU / NVENC 利用率）。\n\n"
            "跑法：\n"
            "  1. 这里点开始（它会先停掉正在跑的「屏幕推流」）；\n"
            "  2. 在 B 机跑  python -m tools.stream_sweep  —— **现在跑、或者先跑着等都行**\n"
            "     （每段开始会公告给 B 机的 UDP 5002，它按公告对齐，不再需要掐时间；\n"
            "      B 机一收到就回执，这里会显示「← B 机已跟上（第 n 段）」。\n"
            "      一条回执都没有 = B 机没在跑，或防火墙挡了 5002）；\n"
            "  3. 把 A 机生成的 perf_push_A.json 拷到 B 机，再跑\n"
            "     python -m tools.stream_sweep --merge perf_push_A.json  出表与结论。\n\n"
            "会反复重启推流（游戏画面会闪几次），跑完不自动开回推流 —— 确认结果后\n"
            "自己在「屏幕推流」卡片上点启动。")
        act.triggered.connect(self._start_push_sweep)
        bar.addAction(act)
        self.act_sweep = act

        act_stop = QAction("停止自检", self)
        act_stop.setToolTip(
            "中断正在跑的推流自检（当前那条 ffmpeg 会被停掉，已测的几段照样落盘）。\n"
            "为什么要有它：一轮 18 条要七八分钟，中途发现参数写错 / 要改候选清单时\n"
            "**必须能停下** —— 否则只能硬关部署台，而那样 ffmpeg 子进程会残留。")
        act_stop.triggered.connect(self._stop_push_sweep)
        act_stop.setEnabled(False)
        bar.addAction(act_stop)
        self.act_sweep_stop = act_stop

        # 设置放**最后**：低频动作往后放，常点的自检留在顺手的位置。
        act_set = QAction("设置…", self)
        act_set.setToolTip(
            "界面字号（**与工作台共用 config/ui.yaml**：这里改了，B 机工作台也是这个字号）、\n"
            "右下角「运行日志」保留多少行。\n\n"
            "字号点确定后立即生效，不用重启。")
        act_set.triggered.connect(self._on_settings)
        bar.addAction(act_set)
        self.act_settings = act_set

    def _on_settings(self):
        """打开设置弹窗。"""
        try:
            dlg = DeploySettingsDialog(self.cfg, on_saved=self._on_settings_saved,
                                       parent=self)
            dlg.exec_()
        except Exception:
            # 槽里抛出的异常在 PyQt5 里会走到 qFatal —— 整个部署台**无征兆消失**
            # （见 gui/worker.py 里 safe_slot 的说明）。部署台工具栏这几处没用那层
            # 保护，所以这里自己兜住，并且把话说到界面上，而不是让它静静死掉。
            traceback.print_exc()
            QMessageBox.warning(self, "设置打不开", traceback.format_exc(limit=3))

    def _on_settings_saved(self):
        """设置弹窗点了确定：把界面同步过来，再走**与改卡片参数同一条**落盘路径。

        为什么不在这里自己 save()：`dcfg.save()` 是整份写，而界面上五张卡片各自
        有内存态 —— 直接写会拿一份过期的 cfg 把卡片刚改的东西盖掉（见 _save_now）。
        """
        from gui import theme            # 与工作台共用 config/ui.yaml
        self.log_pane.set_history(int(self.cfg["log"].get("max_lines")
                                      or LogPane.HISTORY))
        self.cfg["log"]["filter"] = self.log_pane.filter
        self._save_timer.start()         # 300ms 合并，与改参数同一条节拍
        self.log("ui", "设置已更新：界面字号 %d px、日志保留 %d 行"
                 % (theme.load_size(), self.log_pane.HISTORY))

    def _stop_push_sweep(self):
        sw = getattr(self, "_sweep", None)
        if sw is None or not sw.is_alive():
            self.log("ui", "没有正在跑的推流自检")
            self.act_sweep_stop.setEnabled(False)
            return
        sw.stop()
        self.log("ui", "已请求停止推流自检（当前这条 ffmpeg 会停掉，已测的段会落盘）")

    def _start_push_sweep(self):
        """启动推流自检（先停正在跑的推流，避免两路推流混在一起）。"""
        if getattr(self, "_sweep", None) is not None and self._sweep.is_alive():
            QMessageBox.information(self, "正在跑", "推流自检已经在跑了。")
            return
        conf = pp.load()
        mins = len(conf["presets"]) * (conf["seconds"] + conf["settle_seconds"] + 3) / 60.0
        push_running = bool(self.procs.get("push") and self.procs["push"].running())
        r = QMessageBox.question(
            self, "开始推流自检",
            "将按 config/push_presets.json 的 **%d 条候选**逐条换参试推，\n"
            "每条试推 %.0f 秒（前 %.0f 秒丢弃），一轮约 **%.0f 分钟**。\n\n"
            "%s\n\n"
            "期间游戏画面会闪几次（每次都在重启 ffmpeg）。开始吗？"
            % (len(conf["presets"]), conf["seconds"], conf["settle_seconds"], mins,
               ("正在跑的「屏幕推流」会先被停掉。" if push_running else
                "（「屏幕推流」当前没在跑。）")),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if r != QMessageBox.Yes:
            return
        if push_running:
            self.stop_service("push")
        self.log("ui", "推流自检开始：%d 条候选 —— B 机那边请现在跑 tools.stream_sweep"
                 % len(conf["presets"]))
        from deploy.push_sweep import PushSweep
        self._sweep = PushSweep(self.cfg,
                               on_line=lambda t: self.sig_line.emit("ui", t),
                               on_done=lambda rows, path: self.sig_sweep_done.emit(
                                   rows, str(path)))
        self._sweep.start()
        self.act_sweep.setEnabled(False)
        self.act_sweep_stop.setEnabled(True)

    def _on_sweep_done(self, rows, path):
        """自检跑完（在界面线程）：把 A 机侧的结论说清，并提醒去 B 机合并。"""
        self.act_sweep.setEnabled(True)
        self.act_sweep_stop.setEnabled(False)
        bad = [r for r in rows if not r.get("ok", True)]
        slow = [r for r in rows if (r.get("speed") or 0) < 1.0]
        best = max((r for r in rows if (r.get("speed") or 0) >= 1.0),
                   key=lambda r: (r.get("speed") or 0), default=None)
        msg = ["A 机侧跑完 %d 条，记账在：\n  %s" % (len(rows), path), ""]
        if slow:
            msg.append("**A 机跟不上的（speed < 1.00）**：%s"
                       % "、".join("%s(%.2f)" % (r["name"], r["speed"]) for r in slow))
        if bad:
            msg.append("**没起得来的**：%s" % "、".join(r["name"] for r in bad))
        if best:
            msg.append("A 机侧余量最大的：%s（speed %.2f、实际 %.1f fps）"
                       % (best["name"], best["speed"], best.get("out_fps") or -1))
        msg.append("")
        msg.append("延迟那一半在 B 机：\n"
                   "  1. 把上面那个 perf_push_A.json 拷到 B 机；\n"
                   "  2. python -m tools.stream_sweep --merge perf_push_A.json")
        msg.append("跑完不自动开回推流 —— 定下来参数后自己在「屏幕推流」卡片点启动。")
        self.log("ui", "推流自检完成（%d 条）→ %s" % (len(rows), path))
        QMessageBox.information(self, "推流自检完成", "\n".join(msg))

    def _build(self):
        central = QWidget()
        central.setObjectName("Central")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(8)

        root.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(10)

        left = QScrollArea()
        left.setWidgetResizable(True)
        left.setFrameShape(QFrame.NoFrame)
        left.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        lw = QWidget()
        ll = QVBoxLayout(lw)
        ll.setContentsMargins(0, 0, 6, 0)
        ll.setSpacing(8)
        for key in services.ORDER:
            card = ServiceCard(key, self.cfg[key])
            card.changed.connect(self._on_param_changed)
            card.start_clicked.connect(self.start_service)
            card.stop_clicked.connect(self.stop_service)
            self.cards[key] = card
            ll.addWidget(card)
        ll.addStretch(1)
        left.setWidget(lw)

        right = QSplitter(Qt.Vertical)
        self.self_check = SelfCheckPane()
        self.self_check.rerun.connect(self.run_selfcheck)
        self.self_check.apply_link.connect(self.apply_from_link)
        self.log_pane = LogPane()
        # 日志筛选也复现上次选的那个
        i = self.log_pane.cmb.findData(self.cfg["log"].get("filter") or "all")
        if i >= 0:
            self.log_pane.cmb.setCurrentIndex(i)
        # 日志保留行数也复现上次设置的那个（设置弹窗里改的就是它）
        self.log_pane.set_history(int(self.cfg["log"].get("max_lines")
                                      or LogPane.HISTORY))
        right.addWidget(self.self_check)
        right.addWidget(self.log_pane)
        right.setStretchFactor(0, 0)
        right.setStretchFactor(1, 1)
        right.setSizes([300, 460])

        body.addWidget(left, 4)
        body.addWidget(right, 5)
        root.addLayout(body, 1)

        foot = QHBoxLayout()
        foot.setSpacing(8)
        self.lbl_saved = QLabel("")
        self.lbl_saved.setObjectName("CardSub")
        foot.addWidget(self.lbl_saved)
        foot.addStretch(1)
        self.lbl_local = QLabel("")
        self.lbl_local.setObjectName("CardSub")
        foot.addWidget(self.lbl_local)
        root.addLayout(foot)

    def _build_header(self):
        box = QFrame()
        box.setObjectName("Card")
        root = QHBoxLayout(box)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(10)

        col = QVBoxLayout()
        col.setSpacing(2)
        title = QLabel(APP_NAME)
        title.setObjectName("HeadTitle")
        col.addWidget(title)
        self.lbl_flow = QLabel("")
        self.lbl_flow.setObjectName("CardSub")
        col.addWidget(self.lbl_flow)
        root.addLayout(col)
        root.addStretch(1)

        self.chips = {}
        for key in services.ORDER:
            lbl = QLabel("● %s" % services.TITLE[key])
            lbl.setObjectName("Chip")
            lbl.setToolTip(services.SUB[key])
            self.chips[key] = lbl
            root.addWidget(lbl)

        root.addSpacing(10)
        self.btn_all_start = QPushButton("全部启动")
        self.btn_all_start.setObjectName("Primary")
        self.btn_all_start.setToolTip("按部署顺序启动四项（已经在跑的不重复启动）")
        self.btn_all_start.clicked.connect(self.start_all)
        root.addWidget(self.btn_all_start)

        self.btn_all_stop = QPushButton("全部停止")
        self.btn_all_stop.setObjectName("Danger")
        self.btn_all_stop.clicked.connect(self.stop_all)
        root.addWidget(self.btn_all_stop)
        return box

    # ---------------- 配置同步（每次改动都落盘）----------------

    def _on_param_changed(self):
        for key, card in self.cards.items():
            self.cfg[key].update(card.values())
        for card in self.cards.values():
            card.refresh_cmd()
        self._refresh_flow()
        self._save_timer.start()          # 300ms 合并：连点数字框不至于每次都写盘

    def _save_now(self):
        for key, card in self.cards.items():
            self.cfg[key].update(card.values())
        self.cfg["window"] = {"w": self.width(), "h": self.height()}
        self.cfg["log"]["filter"] = self.log_pane.filter
        ok = dcfg.save(self.cfg)
        self.lbl_saved.setText(
            ("已保存 → config/%s" % dcfg.PATH.name) if ok else "保存失败（只读？）")

    def _sync_cards_from_cfg(self):
        """把配置回填到界面（启动时 / 「按 link.yaml 填」之后）。

        做法是**重建卡片**而不是逐个 setValue：串口下拉的候选项、分辨率下拉、
        编辑框里的文本都要跟着配置走，逐个设容易漏；重建 4 张卡片是瞬时的，
        而且「界面 = 配置」这件事一定是真的。
        """
        self._rebuild_cards()

    def _rebuild_cards(self):
        left = self.cards[services.ORDER[0]].parentWidget()
        layout = left.layout() if left is not None else None
        if layout is None:
            return
        for card in self.cards.values():
            layout.removeWidget(card)
            card.setParent(None)
        self.cards = {}
        for i, key in enumerate(services.ORDER):
            card = ServiceCard(key, self.cfg[key])
            card.changed.connect(self._on_param_changed)
            card.start_clicked.connect(self.start_service)
            card.stop_clicked.connect(self.stop_service)
            self.cards[key] = card
            layout.insertWidget(i, card)
        self._refresh_flow()
        self._refresh_states()

    def _refresh_cmd_all(self):
        for card in self.cards.values():
            card.refresh_cmd()

    # ---------------- 状态显示 ----------------

    def _refresh_flow(self):
        push = self.cfg["push"]
        self.lbl_flow.setText(
            "推流目标 udp://%s:%s   ·   分辨率 %sx%s @ %s fps   ·   %s"
            % (push.get("host"), push.get("port"), push.get("width"),
               push.get("height"), push.get("fps"), push.get("encoder")))

    def _refresh_states(self):
        for key, card in self.cards.items():
            proc = self.procs.get(key)
            running = bool(proc and proc.running())
            card.set_running(running, proc.uptime() if running else 0.0)
            chip = self.chips[key]
            chip.setStyleSheet(
                "color: %s;" % (SERVICE_COLOR[key] if running else "#9aa0a6"))
        running = sum(1 for p in self.procs.values() if p.running())
        self.btn_all_stop.setEnabled(running > 0)
        self.btn_all_start.setEnabled(running < len(services.ORDER))

    # ---------------- 启停 ----------------

    def _check_host_before_push(self, cfg):
        """起推流前 ping 一次 B 机，先说一句人话。

        ffmpeg 碰到「本机到 B 机没有路由」只会报 `Error number -10051`，光看那行
        看不出是网络问题（翻译表见 services.explain）。**只提示不拦**：ICMP 有可能
        被防火墙挡住，ping 不通不代表 UDP 发不出去 —— 拦下来反而会挡住能通的情况。
        B 机开着时 ping 是毫秒级，只有它不可达时才会等那 ~0.8 秒。
        """
        try:
            rows = selfcheck.check_host(cfg)
        except Exception:
            return
        for it in rows:
            if it.get("level") == "ok":
                self.log("ui", "推流前检查：%s" % it["title"])
            else:
                self.log("ui", "推流前检查：%s —— %s"
                         % (it["title"], (it.get("detail") or "").splitlines()[0]))

    def _start_one(self, key, quiet=False):
        cfg = self.cfg[key]
        hint = services.missing_hint(key, cfg)
        if hint:
            if quiet:
                self.log("ui", "跳过「%s」：%s" % (services.TITLE[key],
                                                  hint.splitlines()[0]))
            else:
                QMessageBox.warning(self, "先补一下", hint)
            return False

        if key == "push":
            self._check_host_before_push(cfg)

        cmd = services.build_cmd(key, cfg)
        proc = Proc(key, cmd, cwd=dcfg.ROOT,
                    on_line=lambda t, k=key: self.sig_line.emit(k, t),
                    on_exit=lambda c, k=key: self.sig_exit.emit(k, c))
        ok, err = proc.start()
        if not ok:
            self.log("ui", "「%s」启动失败：%s" % (services.TITLE[key], err))
            return False
        self.procs[key] = proc
        self.log(key, "已启动  PID %d" % proc.pid())
        self._refresh_states()
        return True

    def start_service(self, key):
        self._start_one(key)
        self._refresh_states()

    def stop_service(self, key):
        proc = self.procs.get(key)
        if proc is None or not proc.running():
            return
        # 标记「这次退出是我们主动要的」：退出回调在读取线程里，可能比
        # proc.stop() 返回还晚，所以**不能**在这里就清掉（否则会被记成意外退出）。
        # 清的理由在 _on_exit 里。
        self._stopping.add(key)
        self.log(key, "正在停止…")
        ok, msg = proc.stop()
        self.log(key, msg if ok else ("停止有问题：%s" % msg))
        if key == "kbd" and ok:
            # 中继是硬停的，来不及松键 —— 自己补一条 RELEASEALL（见 services）
            _ok, msg2 = services.release_all(self.cfg["kbd"].get("serial"))
            self.log("kbd", msg2)
        self._refresh_states()

    def start_all(self):
        for key in services.ORDER:
            proc = self.procs.get(key)
            if proc is not None and proc.running():
                continue
            self._start_one(key, quiet=True)
        self.log("ui", "全部启动完成（未通过的项见上面的跳过提示）")
        self._refresh_states()

    def stop_all(self):
        for key in reversed(services.ORDER):
            proc = self.procs.get(key)
            if proc is not None and proc.running():
                self.stop_service(key)
        self.log("ui", "全部停止完成")
        self._refresh_states()

    # ---------------- 回调（都在界面线程）----------------

    def _on_line(self, key, text):
        self.log(key, text)

    def _on_exit(self, key, code):
        """子进程结束（在读取线程里发的信号）。"""
        asked = key in self._stopping
        self._stopping.discard(key)
        if asked or code in (0, None):
            self.log(key, "已退出（code %s）" % (code,))
        elif code in (3221225786, -1073741510):     # CTRL_BREAK / Ctrl+C 的退出码
            self.log(key, "已停止")
        else:
            # 把「跑了多久」也写上：一启动就挂 vs 挂了几小时才挂，是完全不同的两类
            # 问题（前者多为配置，后者多为掉线/省电，见 services.explain 的翻译表）。
            proc = self.procs.get(key)
            started = getattr(proc, "started_at", 0.0)
            ran = ("，已运行 %s" % _fmt_dur(time.time() - started)) if started else ""
            self.log(key, "意外退出（code %s%s）—— 上面几行是它的最后输出"
                     % (code, ran))
        # 不用手动清 proc：Proc.running() 走 Popen.poll()，进程退出后自然为假，
        # 清掉 proc 反而会让「再启动」少一个可查的对象。
        self._refresh_states()

    def log(self, key, text):
        for line in str(text).splitlines() or [""]:
            self.log_pane.append(key, line)
        # 已知错误特征 → 跟一句人话。放在唯一的日志入口，服务和界面消息都过这里；
        # 同一条翻译只跟一次，免得 ffmpeg 把同一个错重复报三遍就刷三遍解释。
        hint = services.explain(text)
        if hint and hint not in self._hinted:
            self._hinted.add(hint)
            for ln in hint.splitlines():
                self.log_pane.append("ui", ln)

    # ---------------- 自检 ----------------

    def run_selfcheck(self):
        self.self_check.lbl_when.setText("检查中…")
        cfg = self.cfg
        expect = dcfg.link_expect()
        self.log("ui", "开始环境自检…")

        def work():
            items = selfcheck.run(cfg, expect)
            self.sig_check.emit(items)

        threading.Thread(target=work, daemon=True).start()

    def _on_check(self, items):
        self.self_check.set_items(items)
        bad = [i for i in items if i["level"] != "ok"]
        self.log("ui", "自检完成：%d 项，%d 项需要注意"
                 % (len(items), len(bad)))
        for it in bad:
            self.log("ui", "  [%s] %s" % (it["level"], it["title"]))
        ips = selfcheck.local_ips()
        self.lbl_local.setText("本机 %s → B 机 %s:%s" % (
            ips[0] if ips else "?", self.cfg["push"].get("host"),
            self.cfg["push"].get("port")))

    def apply_from_link(self):
        """把 link.yaml 里的期望值填进部署台（治「两边不一致」）。"""
        lk = dcfg.link_cfg()
        stream = lk.get("stream") or {}
        clock = lk.get("clock_sync") or {}
        kbd = lk.get("kbd") or {}
        push = self.cfg["push"]

        def put(dst, key, val):
            if val not in (None, ""):
                dst[key] = val

        put(push, "host", lk.get("b_host"))
        put(push, "port", stream.get("port"))
        put(push, "width", stream.get("width"))
        put(push, "height", stream.get("height"))
        put(push, "fps", stream.get("fps"))
        put(self.cfg["clock"], "port", clock.get("server_port"))
        put(self.cfg["kbd"], "port", kbd.get("port"))
        put(self.cfg["kbd"], "serial", kbd.get("serial"))
        put(self.cfg["kbd"], "cert", kbd.get("cert"))
        self._rebuild_cards()
        self._save_now()
        self.log("ui", "已按 config/link.yaml 填好（不一致的项已对齐）")
        self.run_selfcheck()

    # ---------------- 关闭 ----------------

    def closeEvent(self, e):
        running = [k for k, p in self.procs.items() if p.running()]
        if running:
            names = "、".join(services.TITLE[k] for k in running)
            r = QMessageBox.question(
                self, "还有服务在跑",
                "这些还在运行：\n  %s\n\n"
                "关闭界面会一起停掉它们（停键盘中继时会顺手给 Pro Micro "
                "补一条 RELEASEALL，把按住的键松开）。\n\n确定退出吗？" % names,
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                e.ignore()
                return
            self.stop_all()
        self._save_now()
        e.accept()


def _install_crash_handlers():
    """让崩溃留下证据，并且**绝不影响启动**。

    **为什么必须显式给文件、还要包 try**：用 pythonw 启动时没有控制台，
    sys.stderr 是 None —— `faulthandler.enable()` 在这种环境下会抛
    RuntimeError("sys.stderr is None")。那一行原本在 main() 里裸着，于是双击
    「被控机部署台.bat」就变成：窗口不出现、没有任何提示、进程秒退（exit 1），
    表现完全是「打不开」。（工作台那边一直是 faulthandler.enable(f) 显式给文件，
    所以没踩到。）

    未捕获异常也写文件：pythonw 下把 traceback 打进 None 等于丢掉。
    """
    try:
        f = open(CRASH_LOG, "a", buffering=1, encoding="utf-8")
        f.write("\n=== 启动 %s ===\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
        faulthandler.enable(f)
    except Exception:
        pass

    def hook(exc_type, exc, tb):
        if _reported:                   # _report_startup_error 已经记过这份堆栈了
            return
        try:
            with open(CRASH_LOG, "a", encoding="utf-8") as fh:
                fh.write("\n=== 未捕获异常 %s ===\n"
                         % time.strftime("%Y-%m-%d %H:%M:%S"))
                traceback.print_exception(exc_type, exc, tb, file=fh)
        except Exception:
            pass
        if sys.stderr is not None:      # pythonw 下是 None，写它只会再抛一次
            traceback.print_exception(exc_type, exc, tb)

    sys.excepthook = hook


def _fmt_dur(sec):
    """秒数 → 「2 时 5 分 / 3 分 12 秒 / 45 秒」（日志里用，别给一堆小数）。"""
    sec = max(0, int(sec))
    if sec >= 3600:
        return "%d 时 %d 分" % (sec // 3600, (sec % 3600) // 60)
    if sec >= 60:
        return "%d 分 %d 秒" % (sec // 60, sec % 60)
    return "%d 秒" % sec


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    try:
        from gui import theme
        theme.apply(app)          # 字号与工作台共用 config/ui.yaml
    except Exception:
        pass

    _install_crash_handlers()

    icon = dcfg.ROOT / "gui" / "icon.ico"
    if icon.exists():
        from PyQt5.QtGui import QIcon
        app.setWindowIcon(QIcon(str(icon)))

    w = DeployWindow()
    w.show()
    return app.exec_()


if __name__ == "__main__":
    # 只接**真正的异常**。`sys.exit()` 在正常关窗时抛的是 SystemExit —— 它属于
    # BaseException 而**不是** Exception。原先写 `except BaseException`，于是每次
    # 正常关掉部署台都会：往 deploy_crash.log 写一条「启动失败 + SystemExit: 0」
    # 并弹一个错误框（日志里那两条就是这么来的）。
    try:
        code = main()
    except Exception:
        _report_startup_error()
        raise
    sys.exit(code)
