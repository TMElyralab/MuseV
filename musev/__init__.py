import os
import logging
import logging.config

# 读取日志配置文件内容
logging.config.fileConfig(os.path.join(os.path.dirname(__file__), "logging.conf"))

# 创建一个日志器logger
logger = logging.getLogger("musev")
