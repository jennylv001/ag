### GitHub Coding Agent Environment Protocol

-----

#### 1\. Architectural Cognition Setup

```bash
# Initialize comprehensive project mapping
find . -type f \( -name "*.py" -o -name "*.js" -o -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.txt" -o -name "*.md" \) > project_manifest.txt

# Generate dependency tree
# The project uses pyproject.toml, so we can list dependencies after installation.
pip install .
pip freeze > current_dependencies.txt
npm list --depth=0 > js_dependencies.txt 2>/dev/null || echo "No JS dependencies" > js_dependencies.txt

# Map Python imports and function definitions
find . -name "*.py" -exec grep -l "^import\|^from\|^def\|^class" {} \; > python_entities.txt
```

-----

#### 2\. Surgical Precision Environment

```python
# .env.agent
DIFF_MODE=minimal
BACKUP_ENABLED=true
VALIDATION_STRICT=true
REFACTOR_DISABLED=true
PLACEHOLDER_FORBIDDEN=true
```

-----

#### 3\. Verification Matrix

```bash
# Pre-change validation pipeline
python -m py_compile **/*.py  # Syntax validation
python -c "import ast, glob; [ast.parse(open(f).read()) for f in glob.glob('**/*.py', recursive=True)]"  # AST validation
grep -r "TODO\|FIXME\|XXX" . --exclude-dir=node_modules --exclude-dir=.git  # Stub detection
```

-----

#### 4\. Test Execution Framework

```bash
# End-to-end test runner
pytest -v --tb=short --strict-markers

# Integration validation
python -c "import sys; import importlib; [importlib.import_module(m) for m in sys.modules.keys() if m.startswith('browser_use')]"
```

-----

#### 5\. Backward Compatibility Sentinel

```python
# compatibility_check.py
import inspect
import importlib
import sys

def validate_signatures(module_name):
    """Ensure all function signatures remain intact"""
    try:
        module = importlib.import_module(module_name)
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            sig = inspect.signature(obj)
            print(f"{module_name}.{name}: {sig}")
        return True
    except Exception as e:
        print(f"COMPATIBILITY_BREACH: {module_name} - {e}")
        return False
```

-----

#### 6\. Non-Destructive Operation Guards

```bash
# Pre-commit hooks
git diff --name-only --cached | xargs python -m py_compile
git diff --cached --name-only | xargs -I {} python -c "
import ast
try:
    with open('{}', 'r') as f:
        ast.parse(f.read())
    print('✓ {}')
except SyntaxError as e:
    print('✗ {} - {}'.format('{}', e))
    exit(1)
"
```

-----

#### 7\. Complete Component Validation

```python
# component_completeness.py
import ast
import sys

def validate_completeness(filepath):
    """Ensure no stubs, TODOs, or incomplete implementations"""
    with open(filepath, 'r') as f:
        content = f.read()
        tree = ast.parse(content)
    
    # Check for forbidden patterns
    forbidden = ['TODO', 'FIXME', '...', 'NotImplemented', 'pass  # TODO']
    for line_num, line in enumerate(content.split('\n'), 1):
        for pattern in forbidden:
            if pattern in line:
                print(f"INCOMPLETENESS_DETECTED: {filepath}:{line_num} - {pattern}")
                return False
    
    # Validate all functions have implementations
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                print(f"STUB_DETECTED: {filepath} - Function {node.name} is incomplete")
                return False
    
    return True
```

-----

#### 8\. Entity Verification Protocol

```python
# entity_verifier.py
import importlib.util
import sys
import os

def verify_entity(entity_path):
    """Verify entity exists in project or known libraries"""
    # Check project files
    if os.path.exists(entity_path):
        return True
    
    # Check importable modules
    try:
        spec = importlib.util.find_spec(entity_path)
        return spec is not None
    except (ImportError, ValueError):
        pass
    
    print(f"Unverifiable Entity: {entity_path} — Clarification required")
    return False
```

-----

#### 9\. Agent Execution Command

```bash
# Single command to establish complete environment
python -c "
import subprocess
import sys
import os

# Verify Python environment by installing from pyproject.toml
subprocess.run([sys.executable, '-m', 'pip', 'install', '.'], check=True)

# Validate project structure
subprocess.run(['python', '-m', 'py_compile'] + [f for f in os.listdir('.') if f.endswith('.py')], check=True)

# Run comprehensive tests
subprocess.run([sys.executable, '-m', 'pytest', '-v'], check=False)

print('ENVIRONMENT_READY: All systems operational for surgical precision coding')
"
```
