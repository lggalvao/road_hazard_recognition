import os
import ast

def extract_functions_from_file(file_path):
    """Extract function and class names (with docstrings) from a Python file."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            node = ast.parse(f.read(), filename=file_path)
        except SyntaxError:
            return []

    functions = []
    for item in ast.walk(node):
        if isinstance(item, ast.FunctionDef):
            functions.append({
                "type": "function",
                "name": item.name,
                "lineno": item.lineno,
                "docstring": ast.get_docstring(item)
            })
        elif isinstance(item, ast.ClassDef):
            functions.append({
                "type": "class",
                "name": item.name,
                "lineno": item.lineno,
                "docstring": ast.get_docstring(item)
            })
    return functions


def extract_functions_from_project(root_dir):
    """Walk through the project directory and extract all functions."""
    all_functions = []

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(subdir, file)
                funcs = extract_functions_from_file(file_path)
                if funcs:
                    all_functions.append({
                        "file": file_path,
                        "definitions": funcs
                    })
    return all_functions


# Example usage
if __name__ == "__main__":
    project_dir = "C:/Projects/hazard_samples_preprocessing"  # <-- change this to your project root
    
    results = extract_functions_from_project(project_dir)

    for file_info in results:
        print(f"\n📄 File: {file_info['file']}")
        for func in file_info["definitions"]:
            print(f"  - [{func['type']}] {func['name']} (line {func['lineno']})")
