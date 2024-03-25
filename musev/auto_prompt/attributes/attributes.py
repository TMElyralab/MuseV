from copy import deepcopy
from typing import List, Tuple, Dict

from mmcm.utils.str_util import has_key_brace


class BaseAttribute2Text(object):
    """
    属性转化为文本的基类，该类作用就是输入属性，转化为描述文本。
    Base class for converting attributes to text which converts attributes to prompt text.
    """

    name = "base_attribute"

    def __init__(self, name: str = None) -> None:
        """这里类实例初始化设置`name`参数，主要是为了便于一些没有提前实现、通过字符串参数实现的新属性。
            Theses class instances are initialized with the `name` parameter to facilitate the implementation of new attributes that are not implemented in advance and are implemented through string parameters.

        Args:
            name (str, optional): _description_. Defaults to None.
        """
        if name is not None:
            self.name = name

    def __call__(self, attributes) -> str:
        raise NotImplementedError


class AttributeIsTextAndName(BaseAttribute2Text):
    """
    属性文本转换功能类，将key和value拼接在一起作为文本.
    class for converting attributes to text which concatenates the key and value together as text.
    """

    name = "attribute_is_text_name"

    def __call__(self, attributes) -> str:
        if attributes == "" or attributes is None:
            return ""
        attributes = attributes.split(",")
        text = ", ".join(
            [
                "{} {}".format(attr, self.name) if attr != "" else ""
                for attr in attributes
            ]
        )
        return text


class AttriributeIsText(BaseAttribute2Text):
    """
    属性文本转换功能类，将value作为文本.
    class for converting attributes to text which only uses the value as text.
    """

    name = "attribute_is_text"

    def __call__(self, attributes: str) -> str:
        if attributes == "" or attributes is None:
            return ""
        attributes = str(attributes)
        attributes = attributes.split(",")
        text = ", ".join(["{}".format(attr) for attr in attributes])
        return text


class MultiAttr2Text(object):
    """将多属性组成的字典转换成完整的文本描述，目前采用简单的前后拼接方式，以`, `作为拼接符号
    class for converting a dictionary of multiple attributes into a complete text description. Currently, a simple front and back splicing method is used, with `, ` as the splicing symbol.

    Args:
        object (_type_): _description_
    """

    def __init__(self, funcs: list, name) -> None:
        """
        Args:
            funcs (list): 继承`BaseAttribute2Text`并实现了`__call__`函数的类. Inherited `BaseAttribute2Text` and implemented the `__call__` function of the class.
            name (_type_): 该多属性的一个名字，可通过该类方便了解对应相关属性都是关于啥的。 name of the multi-attribute, which can be used to easily understand what the corresponding related attributes are about.
        """
        if not isinstance(funcs, list):
            funcs = [funcs]
        self.funcs = funcs
        self.name = name

    def __call__(
        self, dct: dict, ignored_blank_str: bool = False
    ) -> List[Tuple[str, str]]:
        """
        有时候一个属性可能会返回多个文本，如 style cartoon会返回宫崎骏和皮克斯两种风格，采用外积增殖成多个字典。
        sometimes an attribute may return multiple texts, such as style cartoon will return two styles, Miyazaki and Pixar, which are multiplied into multiple dictionaries by the outer product.
        Args:
            dct (dict): 多属性组成的字典，可能有self.funcs关注的属性也可能没有，self.funcs按照各自的名字按需提取关注的属性和值，并转化成文本.
                Dict of multiple attributes, may or may not have the attributes that self.funcs is concerned with. self.funcs extracts the attributes and values of interest according to their respective names and converts them into text.
            ignored_blank_str (bool): 如果某个attr2text返回的是空字符串，是否要过滤掉该属性。默认`False`.
                If the text returned by an attr2text is an empty string, whether to filter out the attribute. Defaults to `False`.
        Returns:
            Union[List[List[Tuple[str, str]]], List[Tuple[str, str]]: 多组多属性文本字典列表. Multiple sets of multi-attribute text dictionaries.
        """
        attrs_lst = [[]]
        for func in self.funcs:
            if func.name in dct:
                attrs = func(dct[func.name])
                if isinstance(attrs, str):
                    for i in range(len(attrs_lst)):
                        attrs_lst[i].append((func.name, attrs))
                else:
                    # 一个属性可能会返回多个文本
                    n_attrs = len(attrs)
                    new_attrs_lst = []
                    for n in range(n_attrs):
                        attrs_lst_cp = deepcopy(attrs_lst)
                        for i in range(len(attrs_lst_cp)):
                            attrs_lst_cp[i].append((func.name, attrs[n]))
                        new_attrs_lst.extend(attrs_lst_cp)
                    attrs_lst = new_attrs_lst

        texts = [
            [
                (attr, text)
                for (attr, text) in attrs
                if not (text == "" and ignored_blank_str)
            ]
            for attrs in attrs_lst
        ]
        return texts


def format_tuple_texts(template: str, texts: Tuple[str, str]) -> str:
    """使用含有"{}" 的模板对多属性文本元组进行拼接，形成新文本
        concatenate multiple attribute text tuples using a template containing "{}" to form a new text
    Args:
        template (str):
        texts (Tuple[str, str]): 多属性文本元组. multiple attribute text tuples

    Returns:
        str: 拼接后的新文本, merged new text
    """
    merged_text = ", ".join([text[1] for text in texts if text[1] != ""])
    merged_text = template.format(merged_text)
    return merged_text


def format_dct_texts(template: str, texts: Dict[str, str]) -> str:
    """使用含有"{key}" 的模板对多属性文本字典进行拼接，形成新文本
        concatenate multiple attribute text dictionaries using a template containing "{key}" to form a new text
    Args:
        template (str):
        texts (Tuple[str, str]): 多属性文本字典. multiple attribute text dictionaries

    Returns:
        str: 拼接后的新文本, merged new text
    """
    merged_text = template.format(**texts)
    return merged_text


def merge_multi_attrtext(texts: List[Tuple[str, str]], template: str = None) -> str:
    """对多属性文本元组进行拼接，形成新文本。
        如果`template`含有{key}，则根据key来取值；
        如果`template`有且只有1个{}，则根据先后顺序对texts中的值进行拼接。

        concatenate multiple attribute text tuples to form a new text.
        if `template` contains {key}, the value is taken according to the key;
        if `template` contains only one {}, the values in texts are concatenated in order.
    Args:
        texts (List[Tuple[str, str]]): Tuple[str, str]第一个str是属性名，第二个str是属性转化的文本.
            Tuple[str, str] The first str is the attribute name, and the second str is the text of the attribute conversion.
        template (str, optional): template . Defaults to None.

    Returns:
        str: 拼接后的新文本, merged new text
    """
    if not isinstance(texts, List):
        texts = [texts]
    if template is None or template == "":
        template = "{}"
    if has_key_brace(template):
        texts = {k: v for k, v in texts}
        merged_text = format_dct_texts(template, texts)
    else:
        merged_text = format_tuple_texts(template, texts)
    return merged_text


class PresetMultiAttr2Text(MultiAttr2Text):
    """预置了多种关注属性转换的类，方便维护
    class for multiple attribute conversion with multiple attention attributes preset for easy maintenance

    """

    preset_attributes = []

    def __init__(
        self, funcs: List = None, use_preset: bool = True, name: str = "preset"
    ) -> None:
        """虽然预置了关注的属性列表和转换类，但也允许定义示例时，进行更新。
        注意`self.preset_attributes`的元素只是类名字，以便减少实例化的资源消耗。而funcs是实例化后的属性转换列表。

        Although the list of attention attributes and conversion classes is preset, it is also allowed to be updated when defining an instance.
        Note that the elements of `self.preset_attributes` are only class names, in order to reduce the resource consumption of instantiation. And funcs is a list of instantiated attribute conversions.

        Args:
            funcs (List, optional): list of funcs . Defaults to None.
            use_preset (bool, optional): _description_. Defaults to True.
            name (str, optional): _description_. Defaults to "preset".
        """
        if use_preset:
            preset_funcs = self.preset()
        else:
            preset_funcs = []
        if funcs is None:
            funcs = []
        if not isinstance(funcs, list):
            funcs = [funcs]
        funcs_names = [func.name for func in funcs]
        preset_funcs = [
            preset_func
            for preset_func in preset_funcs
            if preset_func.name not in funcs_names
        ]
        funcs = funcs + preset_funcs
        super().__init__(funcs, name)

    def preset(self):
        funcs = [cls() for cls in self.preset_attributes]
        return funcs
