from abc import ABC, abstractmethod

from .frame import Frame


class FrameSource(ABC):
    """统一画面源抽象。

    实现：PyAVSource（实时 UDP/SRT/RTSP）、FileSource（离线回放）。
    下游（感知/决策）只依赖本接口，切换协议零成本。
    """

    @abstractmethod
    def open(self) -> None:
        ...

    @abstractmethod
    def read(self) -> Frame | None:
        """返回下一帧；流结束时返回 None。"""
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @property
    def size(self) -> tuple[int, int] | None:
        return None

    @property
    def fps(self) -> float | None:
        return None

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
