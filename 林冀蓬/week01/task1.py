import sys
import platform
import subprocess


def get_installed_packages():
    """
    获取已安装的Python包列表（替代废弃的pkg_resources）
    返回格式：{包名: 版本号}
    """
    installed_packages = {}
    try:
        # 使用pip list命令获取包信息（跨平台通用）
        result = subprocess.check_output(
            [sys.executable, "-m", "pip", "list", "--format=freeze"],
            text=True,
            encoding="utf-8",
            errors="ignore"
        )
        # 解析输出结果
        for line in result.strip().split('\n'):
            if '==' in line:
                pkg_name, pkg_version = line.split('==', 1)
                installed_packages[pkg_name.lower()] = pkg_version
    except subprocess.CalledProcessError:
        print("⚠️  获取包列表失败：pip命令执行出错")
    except Exception as e:
        print(f"⚠️  获取包列表异常：{str(e)}")
    return installed_packages


def print_environment_info():
    print("---系统和Python环境---")
    # 系统信息
    print(f"操作系统： {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python 版本：{sys.version.split()[0]}")  # 简化Python版本输出

    # 修复：改用更通用的方式获取Python安装路径
    print(f"Python 安装路径：{sys.prefix}")  # 替换sysconfig.get_path('prefix')
    print(f"当前工作目录： {sys.path[0]}")
    print("-" * 50)

    # 获取并打印已安装的包列表
    print("---已安装的Python包（前20个）---")
    installed_pkgs = get_installed_packages()
    if installed_pkgs:
        # 按包名排序，只打印前20个避免输出过长
        sorted_pkgs = sorted(installed_pkgs.items())[:20]
        for pkg_name, pkg_version in sorted_pkgs:
            print(f"{pkg_name:<20} {pkg_version}")
        # 打印包总数
        print(f"\n📦 已安装包总数：{len(installed_pkgs)}")
    else:
        print("❌ 未获取到已安装包信息")
    print("-" * 50)


if __name__ == "__main__":
    print_environment_info()