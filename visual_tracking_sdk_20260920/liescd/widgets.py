"""Compact display without rounding the stored tracking tolerance."""
from PySide6.QtWidgets import QDoubleSpinBox


class MotionToleranceSpinBox(QDoubleSpinBox):
    def textFromValue(self, value):
        return self.locale().toString(value, 'f', 3)

    def valueFromText(self, text):
        displayed = self.prefix() + self.textFromValue(self.value()) + self.suffix()
        if text.strip() in (displayed.strip(), self.textFromValue(self.value())):
            return self.value()
        return super().valueFromText(text)
