import ast

def read_config(filename):
    result = []
    with open(filename, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            type_name, funcs = line.strip().split(':', 1)
            func_list = [fn.strip() for fn in funcs.split(',') if fn.strip()]
            result.append((type_name.strip(), func_list))
    return result

class CallAnalyzer(ast.NodeVisitor):
    def __init__(self, config):
        self.config = config  # List of (type_name, [func_names])
        self.matches = []

    def visit_FunctionDef(self, node):
        # Check if this function is inside a class
        class_name = None
        if hasattr(node, 'parent') and isinstance(node.parent, ast.ClassDef):
            class_name = node.parent.name
        else:
            class_name = "Global"

        for type_name, func_names in self.config:
            if type_name == class_name:
                # Check if this function calls any target function
                called = set()
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            fname = child.func.id
                        elif isinstance(child.func, ast.Attribute):
                            fname = child.func.attr
                        else:
                            continue
                        if fname in func_names:
                            called.add(fname)
                for fname in called:
                    self.matches.append({
                        "type": class_name,
                        "func": node.name,
                        "called": fname,
                        "lineno": node.lineno
                    })
        self.generic_visit(node)
    def visit_Module(self, node):
        # 检查全局作用域的调用
        for type_name, func_names in self.config:
            if type_name == "Global":
                called = set()
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            fname = child.func.id
                        elif isinstance(child.func, ast.Attribute):
                            fname = child.func.attr
                        else:
                            continue
                        if fname in func_names:
                            called.add((fname, child.lineno))
                for fname, lineno in called:
                    self.matches.append({
                        "type": "Global",
                        "func": "<module>",
                        "called": fname,
                        "lineno": lineno
                    })
        self.generic_visit(node)

def add_parent_links(tree):
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

def analyze_file(filename, config):
    with open(filename, 'r', encoding='utf-8') as f:
        source = f.read()
    tree = ast.parse(source, filename)
    add_parent_links(tree)
    analyzer = CallAnalyzer(config)
    analyzer.visit(tree)
    return analyzer.matches

if __name__ == "__main__":
    # Example usage
    config = read_config("target_func")
    # 假设分析 foo.py
    matches = analyze_file("f1.py", config)
    for m in matches:
        print("="*70)
        print(f"Tag:\n{m['type']}\n")
        print(f"Location:\n{m['func']} (line {m['lineno']})\n")
        print(f"Function Name:\n{m['called']}\n")
        print("="*70)