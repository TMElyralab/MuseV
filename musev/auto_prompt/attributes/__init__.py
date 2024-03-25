from ...utils.register import Register

AttrRegister = Register(registry_name="attributes")

# must import like bellow to ensure that each class is registered with AttrRegister:
from .human import *
from .render import *
from .style import *
