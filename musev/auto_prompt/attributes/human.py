from copy import deepcopy
import numpy as np
import random
import json

from .attributes import (
    MultiAttr2Text,
    AttriributeIsText,
    AttributeIsTextAndName,
    PresetMultiAttr2Text,
)
from .style import Style
from .render import Render
from . import AttrRegister


__all__ = [
    "Age",
    "Sex",
    "Singing",
    "Country",
    "Lighting",
    "Headwear",
    "Eyes",
    "Irises",
    "Hair",
    "Skin",
    "Face",
    "Smile",
    "Expression",
    "Clothes",
    "Nose",
    "Mouth",
    "Beard",
    "Necklace",
    "KeyWords",
    "InsightFace",
    "Caption",
    "Env",
    "Decoration",
    "Festival",
    "SpringHeadwear",
    "SpringClothes",
    "Animal",
]


@AttrRegister.register
class Sex(AttriributeIsText):
    name = "sex"

    def __init__(self, name: str = None) -> None:
        super().__init__(name)


@AttrRegister.register
class Headwear(AttriributeIsText):
    name = "headwear"

    def __init__(self, name: str = None) -> None:
        super().__init__(name)


@AttrRegister.register
class Expression(AttriributeIsText):
    name = "expression"

    def __init__(self, name: str = None) -> None:
        super().__init__(name)


@AttrRegister.register
class KeyWords(AttriributeIsText):
    name = "keywords"

    def __init__(self, name: str = None) -> None:
        super().__init__(name)


@AttrRegister.register
class Singing(AttriributeIsText):
    def __init__(self, name: str = "singing") -> None:
        super().__init__(name)


@AttrRegister.register
class Country(AttriributeIsText):
    name = "country"

    def __init__(self, name: str = None) -> None:
        super().__init__(name)


@AttrRegister.register
class Clothes(AttriributeIsText):
    name = "clothes"

    def __init__(self, name: str = None) -> None:
        super().__init__(name)


@AttrRegister.register
class Age(AttributeIsTextAndName):
    name = "age"

    def __init__(self, name: str = None) -> None:
        super().__init__(name)

    def __call__(self, attributes: str) -> str:
        if not isinstance(attributes, str):
            attributes = str(attributes)
        attributes = attributes.split(",")
        text = ", ".join(
            ["{}-year-old".format(attr) if attr != "" else "" for attr in attributes]
        )
        return text


@AttrRegister.register
class Eyes(AttributeIsTextAndName):
    name = "eyes"

    def __init__(self, name: str = None) -> None:
        super().__init__(name)


@AttrRegister.register
class Hair(AttributeIsTextAndName):
    name = "hair"

    def __init__(self, name: str = None) -> None:
        super().__init__(name)


@AttrRegister.register
class Background(AttributeIsTextAndName):
    name = "background"

    def __init__(self, name: str = None) -> None:
        super().__init__(name)


@AttrRegister.register
class Skin(AttributeIsTextAndName):
    name = "skin"

    def __init__(self, name: str = None) -> None:
        super().__init__(name)


@AttrRegister.register
class Face(AttributeIsTextAndName):
    name = "face"

    def __init__(self, name: str = None) -> None:
        super().__init__(name)


@AttrRegister.register
class Smile(AttributeIsTextAndName):
    name = "smile"

    def __init__(self, name: str = None) -> None:
        super().__init__(name)


@AttrRegister.register
class Nose(AttributeIsTextAndName):
    name = "nose"

    def __init__(self, name: str = None) -> None:
        super().__init__(name)


@AttrRegister.register
class Mouth(AttributeIsTextAndName):
    name = "mouth"

    def __init__(self, name: str = None) -> None:
        super().__init__(name)


@AttrRegister.register
class Beard(AttriributeIsText):
    name = "beard"

    def __init__(self, name: str = None) -> None:
        super().__init__(name)


@AttrRegister.register
class Necklace(AttributeIsTextAndName):
    name = "necklace"

    def __init__(self, name: str = None) -> None:
        super().__init__(name)


@AttrRegister.register
class Irises(AttributeIsTextAndName):
    name = "irises"

    def __init__(self, name: str = None) -> None:
        super().__init__(name)


@AttrRegister.register
class Lighting(AttributeIsTextAndName):
    name = "lighting"

    def __init__(self, name: str = None) -> None:
        super().__init__(name)


PresetPortraitAttributes = [
    Age,
    Sex,
    Singing,
    Country,
    Lighting,
    Headwear,
    Eyes,
    Irises,
    Hair,
    Skin,
    Face,
    Smile,
    Expression,
    Clothes,
    Nose,
    Mouth,
    Beard,
    Necklace,
    Style,
    KeyWords,
    Render,
]


class PortraitMultiAttr2Text(PresetMultiAttr2Text):
    preset_attributes = PresetPortraitAttributes

    def __init__(self, funcs: list = None, use_preset=True, name="portrait") -> None:
        super().__init__(funcs, use_preset, name)


@AttrRegister.register
class InsightFace(AttriributeIsText):
    name = "insight_face"
    face_render_dict = {
        "boy": "handsome,elegant",
        "girl": "gorgeous,kawaii,colorful",
    }
    key_words = "delicate face,beautiful eyes"

    def __call__(self, attributes: str) -> str:
        """将insight faces 检测的结果转化成prompt
            convert the results of insight faces detection to prompt
        Args:
            face_list (_type_): _description_

        Returns:
            _type_: _description_
        """
        attributes = json.loads(attributes)
        face_list = attributes["info"]
        if len(face_list) == 0:
            return ""

        if attributes["image_type"] == "body":
            for face in face_list:
                if "black" in face and face["black"]:
                    return "african,dark skin"
            return ""

        gender_dict = {"girl": 0, "boy": 0}
        face_render_list = []
        black = False

        for face in face_list:
            if face["ratio"] < 0.02:
                continue

            if face["gender"] == 0:
                gender_dict["girl"] += 1
                face_render_list.append(self.face_render_dict["girl"])
            else:
                gender_dict["boy"] += 1
                face_render_list.append(self.face_render_dict["boy"])

            if "black" in face and face["black"]:
                black = True

        if len(face_render_list) == 0:
            return ""
        elif len(face_render_list) == 1:
            solo = True
        else:
            solo = False

        gender = ""
        for g, num in gender_dict.items():
            if num > 0:
                if gender:
                    gender += ", "
                gender += "{}{}".format(num, g)
                if num > 1:
                    gender += "s"

        face_render_list = ",".join(face_render_list)
        face_render_list = face_render_list.split(",")
        face_render = list(set(face_render_list))
        face_render.sort(key=face_render_list.index)
        face_render = ",".join(face_render)
        if gender_dict["girl"] == 0:
            face_render = "male focus," + face_render

        insightface_prompt = "{},{},{}".format(gender, face_render, self.key_words)

        if solo:
            insightface_prompt += ",solo"
        if black:
            insightface_prompt = "african,dark skin," + insightface_prompt

        return insightface_prompt


@AttrRegister.register
class Caption(AttriributeIsText):
    name = "caption"


@AttrRegister.register
class Env(AttriributeIsText):
    name = "env"
    envs_list = [
        "east asian architecture",
        "fireworks",
        "snow, snowflakes",
        "snowing, snowflakes",
    ]

    def __call__(self, attributes: str = None) -> str:
        if attributes != "" and attributes != " " and attributes is not None:
            return attributes
        else:
            return random.choice(self.envs_list)


@AttrRegister.register
class Decoration(AttriributeIsText):
    name = "decoration"

    def __init__(self, name: str = None) -> None:
        self.decoration_list = [
            "chinese knot",
            "flowers",
            "food",
            "lanterns",
            "red envelop",
        ]
        super().__init__(name)

    def __call__(self, attributes: str = None) -> str:
        if attributes != "" and attributes != " " and attributes is not None:
            return attributes
        else:
            return random.choice(self.decoration_list)


@AttrRegister.register
class Festival(AttriributeIsText):
    name = "festival"
    festival_list = ["new year"]

    def __init__(self, name: str = None) -> None:
        super().__init__(name)

    def __call__(self, attributes: str = None) -> str:
        if attributes != "" and attributes != " " and attributes is not None:
            return attributes
        else:
            return random.choice(self.festival_list)


@AttrRegister.register
class SpringHeadwear(AttriributeIsText):
    name = "spring_headwear"
    headwear_list = ["rabbit ears", "rabbit ears, fur hat"]

    def __call__(self, attributes: str = None) -> str:
        if attributes != "" and attributes != " " and attributes is not None:
            return attributes
        else:
            return random.choice(self.headwear_list)


@AttrRegister.register
class SpringClothes(AttriributeIsText):
    name = "spring_clothes"
    clothes_list = [
        "mittens,chinese clothes",
        "mittens,fur trim",
        "mittens,red scarf",
        "mittens,winter clothes",
    ]

    def __call__(self, attributes: str = None) -> str:
        if attributes != "" and attributes != " " and attributes is not None:
            return attributes
        else:
            return random.choice(self.clothes_list)


@AttrRegister.register
class Animal(AttriributeIsText):
    name = "animal"
    animal_list = ["rabbit", "holding rabbits"]

    def __call__(self, attributes: str = None) -> str:
        if attributes != "" and attributes != " " and attributes is not None:
            return attributes
        else:
            return random.choice(self.animal_list)
