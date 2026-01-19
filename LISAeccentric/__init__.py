# 文件路径: LISAeccentric/__init__.py

# 1. 从同级目录的 core.py 中导入核心类
# 注意：这里假设你已经把那个很长的脚本改名为 core.py
try:
    from .core import LISAeccentric as _CoreEngine
    from .core import CompactBinary
except ImportError as e:
    # 错误提示：帮助调试路径问题
    raise ImportError(f"LISAeccentric package initialization failed. Could not import 'core.py'.\nDetails: {e}")

# 2. 【自动实例化】
# 在包被导入时，悄悄创建一个实例。
# 这样，外界不需要再写 tool = LISAeccentric()，我们帮他做好了。
_default_instance = _CoreEngine()

# 3. 【挂载功能模块】
# 将实例中的子模块赋值给包的顶层变量。
# 这样用户就可以用 "LISAeccentric.GN" 访问，而不是 "LISAeccentric._default_instance.GN"
GN = _default_instance.GN
GC = _default_instance.GC
Field = _default_instance.Field
Waveform = _default_instance.Waveform

# 4. 【暴露数据类】
# 允许用户直接使用 LISAeccentric.CompactBinary(...)
CompactBinary = CompactBinary

# 5. 定义包的公共接口
__all__ = [
    'GN',
    'GC',
    'Field',
    'Waveform',
    'CompactBinary'
]

# 打印初始化成功信息 (可选，调试用)
# print("LISAeccentric package initialized successfully.")