"""
基于gplearn中function函数进行了改进，用于定义公式树中的算子类_Function
"""
class _Function(object):
    def __init__(self, function, name, arity, is_ts=False):
        self.function = function
        self.name = name
        self.arity = arity

        # 新增参数
        self.is_ts = is_ts  # bool, 代表此函数是否为时间序列函数，默认为False
        self.d = 3  # int, 时间序列回滚周期，若为时间序列函数则需要重设此参数

    def __call__(self, *args):
        if not self.is_ts:
            return self.function(*args)
        else:
            if self.d == 0:
                raise AttributeError("Please reset attribute 'd'")
            else:
                return self.function(*args, self.d)

    # 新增重设参数d的方法
    def set_d(self, d):
        self.d = d


