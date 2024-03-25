from mmcm.utils.str_util import has_key_brace

from .human import PortraitAttr2PromptTemplate
from .attributes.attr2template import (
    KeywordMultiAttr2PromptTemplate,
    OnlySpacePromptTemplate,
)


def get_template_by_name(template: str, name: str = None):
    """根据 template_name 确定 prompt 生成器类
        choose prompt generator class according to template_name
    Args:
        name (str): template 的名字简称，便于指定. template name abbreviation, for easy reference

    Raises:
        ValueError: ValueError: 如果name不在支持的列表中，则报错. if name is not in the supported list, an error is reported.

    Returns:
        MultiAttr2PromptTemplate: 能够将任务字典转化为提词的 实现了__call__功能的类. class that can convert task dictionaries into prompts and implements the __call__ function

    """
    if template == "" or template is None:
        template = OnlySpacePromptTemplate(template=template)
    elif has_key_brace(template):
        # if has_key_brace(template):
        template = KeywordMultiAttr2PromptTemplate(template=template)
    else:
        if name == "portrait":
            template = PortraitAttr2PromptTemplate(templates=template)
        else:
            raise ValueError(
                "PresetAttr2PromptTemplate only support one of [portrait], but given {}".format(
                    name
                )
            )
    return template
