"""负责按照人相关的属性转化成提词
"""
from typing import List

from .attributes.human import PortraitMultiAttr2Text
from .attributes.attributes import BaseAttribute2Text
from .attributes.attr2template import MultiAttr2PromptTemplate


class PortraitAttr2PromptTemplate(MultiAttr2PromptTemplate):
    """可以将任务字典转化为形象提词模板类
        template class for converting task dictionaries into image prompt templates
    Args:
        MultiAttr2PromptTemplate (_type_): _description_
    """

    templates = "a portrait of {}"

    def __init__(
        self, templates: str = None, attr2text: List = None, name: str = "portrait"
    ) -> None:
        """

        Args:
            templates (str, optional): 形象提词模板，若为None，则使用默认的类属性. Defaults to None.
                portrait prompt template, if None, the default class attribute is used.
            attr2text (List, optional): 形象类需要新增、更新的属性列表，默认使用PortraitMultiAttr2Text中定义的形象属性. Defaults to None.
                the list of attributes that need to be added or updated in the image class, by default, the image attributes defined in PortraitMultiAttr2Text are used.
            name (str, optional): 该形象类的名字. Defaults to "portrait".
                class name of this class instance
        """
        if (
            attr2text is None
            or isinstance(attr2text, list)
            or isinstance(attr2text, BaseAttribute2Text)
        ):
            attr2text = PortraitMultiAttr2Text(funcs=attr2text)
        if templates is None:
            templates = self.templates
        super().__init__(templates, attr2text, name=name)
