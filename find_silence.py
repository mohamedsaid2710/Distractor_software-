import os
import ast

def check_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                # Is it a bare except or except Exception?
                is_bare = True
                if node.type is not None:
                    if isinstance(node.type, ast.Name) and getattr(node.type, 'id', '') == 'Exception':
                        is_bare = True
                    else:
                        is_bare = False
                
                if is_bare:
                    if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                        print(f"Silent Pass: {path}:{node.lineno}")
                    elif len(node.body) == 1 and isinstance(node.body[0], ast.Continue):
                        print(f"Silent Continue: {path}:{node.lineno}")
                    elif len(node.body) == 1 and isinstance(node.body[0], ast.Return):
                        print(f"Silent Return: {path}:{node.lineno}")

    except Exception:
        pass

for root, _, files in os.walk('/mnt/c/Users/moham/Desktop/maze_automate'):
    if '.venv' in root or '.git' in root or '__pycache__' in root:
        continue
    for file in files:
        if file.endswith('.py'):
            check_file(os.path.join(root, file))
