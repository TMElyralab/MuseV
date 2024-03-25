from mmcm.utils.util import flatten

from .attributes import BaseAttribute2Text
from . import AttrRegister

__all__ = ["Render"]

RenderMap = {
    "Epic": "artstation, epic environment, highly detailed, 8k, HD",
    "HD": "8k, highly detailed",
    "EpicHD": "hyper detailed, beautiful lighting, epic environment, octane render, cinematic, 8k",
    "Digital": "detailed illustration, crisp lines, digital art, 8k, trending on artstation",
    "Unreal1": "artstation, concept art, smooth, sharp focus, illustration, unreal engine 5, 8k",
    "Unreal2": "concept art, octane render, artstation, epic environment, highly detailed, 8k",
}


@AttrRegister.register
class Render(BaseAttribute2Text):
    name = "render"

    def __init__(self, name: str = None) -> None:
        super().__init__(name)

    def __call__(self, attributes: str) -> str:
        if attributes == "" or attributes is None:
            return ""
        attributes = attributes.split(",")
        render = [RenderMap[attr] for attr in attributes if attr in RenderMap]
        render = flatten(render, ignored_iterable_types=[str])
        if len(render) == 1:
            render = render[0]
        return render
