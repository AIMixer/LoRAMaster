import logging
import os
from logging.handlers import TimedRotatingFileHandler

# 日志目录
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 日志文件路径（按日期分割）
log_file = os.path.join(LOG_DIR, "lora.log")

# 创建 logger
logger = logging.getLogger("LoRAMaster")
logger.setLevel(logging.DEBUG)  # 总开关

# 日志格式
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 文件 Handler（每天生成一个日志文件，最多保留7天）
file_handler = TimedRotatingFileHandler(
    log_file, when="midnight", interval=1, backupCount=7, encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)  # 文件保存所有级别
file_handler.setFormatter(formatter)

# 控制台 Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 控制台只输出 INFO 及以上
console_handler.setFormatter(formatter)

# 避免重复添加 handler
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# 防止子 logger 重复输出
logger.propagate = False
