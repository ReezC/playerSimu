"""共用的基础控件。

**为什么需要这些子类**

    QSpinBox / QDoubleSpinBox / QComboBox / QSlider 默认「指针悬停就响应滚轮」。
    在滚动区里这是灾难：想翻页，指针顺路划过参数框，数字就被改了，
    而且改完没有任何提示。

    更隐蔽的是「聚焦残留」：点过一次数字框后焦点还留在它上面，
    之后再滚页面，滚轮仍会被当成"在调这个参数"—— 表现为
    「没聚焦时不误改了，但只要点过一下就又开始了」。

**规则：这些控件永远不响应滚轮**

    滚轮专心滚页面 / 缩放画布；参数只用上下箭头、下拉框、键盘来改。
    这些控件都有明确的非滚轮调节方式，禁用滚轮不损失任何操作效率，
    反而彻底根除了"到底是谁在抢滚轮"的纠结。
"""

from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox, QSlider, QSpinBox


class _NoWheel:
    """滚轮事件永远让给父级（滚动区 / 画布）。"""

    def wheelEvent(self, ev):
        ev.ignore()


class NoWheelSpinBox(_NoWheel, QSpinBox):
    pass


class NoWheelDoubleSpinBox(_NoWheel, QDoubleSpinBox):
    pass


class NoWheelComboBox(_NoWheel, QComboBox):
    pass


class NoWheelSlider(_NoWheel, QSlider):
    pass
