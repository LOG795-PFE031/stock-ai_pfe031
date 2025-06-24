import importlib
from pathlib import Path

"""
Automatically imports all trainer modules in this package, 
excluding 'base_model.py' and 'trainer_factory.py'. 

This triggers the registration of a trainer with the trainer factory.
"""

# Get all .py files in the models directory
current_dir = Path(__file__).parent
for file_path in current_dir.glob("*.py"):
    if file_path.name not in ["__init__.py", "base_model.py", "trainer_factory.py"]:
        module_name = file_path.stem
        # This import the module, triggering registration
        importlib.import_module(f".{module_name}", package="training.trainers")
