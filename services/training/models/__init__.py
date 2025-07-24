import importlib
from pathlib import Path

"""
Automatically imports all models modules in this package, 
excluding 'base_model.py'. 

This triggers the registration of a model with the model factory.
"""

# Get all .py files in the models directory
current_dir = Path(__file__).parent
for file_path in current_dir.glob("*.py"):
    if file_path.name not in ["__init__.py", "base_model.py"]:
        module_name = file_path.stem
        # This import the module, triggering registration
        importlib.import_module(f".{module_name}", package=__package__)
