import subprocess
import sys
from pathlib import Path


def test_evaluator_imports_only_release_source():
    root = Path(__file__).resolve().parents[1]
    code = """
import sys
from pathlib import Path
from scripts.eval.seekvln_auxthink_eval import SeekVLNDirectEvaluator
root = Path(sys.argv[1]).resolve()
names = []
for name, module in list(sys.modules.items()):
    if name.startswith(('llava', 'vlnce_baselines', 'habitat_extensions', 'evaluation', 'scripts.eval')):
        file = getattr(module, '__file__', None)
        if file:
            assert Path(file).resolve().is_relative_to(root), (name, file)
            names.append(name)
assert not any(name == 'verl' or name.startswith(('verl.', 'ray.', 'llava.trl.trainer')) for name in sys.modules)
print('RELEASE_IMPORTS_VERIFIED', len(names))
"""
    subprocess.run([sys.executable, "-B", "-c", code, str(root)], check=True)
