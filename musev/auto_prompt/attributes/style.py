from .attributes import AttriributeIsText
from . import AttrRegister

__all__ = ["Style"]


@AttrRegister.register
class Style(AttriributeIsText):
    name = "style"

    def __init__(self, name: str = None) -> None:
        super().__init__(name)
