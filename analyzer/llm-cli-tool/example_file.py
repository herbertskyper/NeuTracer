# 示例Python代码 - 计算器类


class Calculator:
    """一个简单的计算器类，支持基本数学运算"""

    def __init__(self):
        self.history = []

    def add(self, a, b):
        """加法运算"""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def subtract(self, a, b):
        """减法运算"""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result

    def multiply(self, a, b):
        """乘法运算"""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

    def divide(self, a, b):
        """除法运算"""
        if b == 0:
            raise ValueError("除数不能为零")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result

    def get_history(self):
        """获取计算历史"""
        return self.history

    def clear_history(self):
        """清空计算历史"""
        self.history = []


# 使用示例
if __name__ == "__main__":
    calc = Calculator()

    print("计算器测试:")
    print(f"5 + 3 = {calc.add(5, 3)}")
    print(f"10 - 4 = {calc.subtract(10, 4)}")
    print(f"6 * 7 = {calc.multiply(6, 7)}")
    print(f"15 / 3 = {calc.divide(15, 3)}")

    print("\n计算历史:")
    for operation in calc.get_history():
        print(operation)
