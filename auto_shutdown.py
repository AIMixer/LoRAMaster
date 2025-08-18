import os
import platform
import ctypes
from logger import logger

def is_admin() -> bool:
    """
    判断当前程序是否以管理员/Root 权限运行
    Windows: 调用系统 API 检查管理员权限
    Linux/macOS: 检查 uid 是否为 0
    """
    system = platform.system()
    if system == "Windows":
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False
    else:
        return os.geteuid() == 0


def shutdown(delay_seconds: int = 300, restart: bool = False):
    """
    执行关机或重启操作

    :param delay_seconds: 延迟关机/重启的秒数 (0 = 立即)
    :param restart: True 表示重启, False 表示关机
    """
    if not is_admin():
        logger.info("❌ 当前没有管理员权限，无法执行关机/重启命令")
        return

    system = platform.system()
    if system == "Windows":
        if restart:
            os.system(f"shutdown /r /t {delay_seconds}")
        else:
            os.system(f"shutdown /s /t {delay_seconds}")
    elif system in ("Linux", "Darwin"):  # Darwin = macOS
        minutes = max(1, delay_seconds // 60) if delay_seconds > 0 else 0
        if restart:
            if minutes > 0:
                os.system(f"shutdown -r +{minutes}")
            else:
                os.system("shutdown -r now")
        else:
            if minutes > 0:
                os.system(f"shutdown -h +{minutes}")
            else:
                os.system("shutdown -h now")
    else:
        logger.info("❌ 不支持的系统关机命令")


if __name__ == "__main__":
    # 示例：立即关机
    shutdown()

    # 示例：60秒后关机
    # shutdown(delay_seconds=60)

    # 示例：立即重启
    # shutdown(restart=True)
