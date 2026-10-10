import os
import subprocess
import sys


def test_minimal_vlnce_import_skips_training_only_modules():
    code = """
import sys
import vlnce_baselines
assert 'vlnce_baselines.common.environments' in sys.modules
assert 'vlnce_baselines.cma_trainer' not in sys.modules
assert 'vlnce_baselines.dagger_trainer' not in sys.modules
"""
    env = dict(os.environ)
    env["SEEKVLN_MINIMAL_VLNCE_IMPORT"] = "1"
    evaluation = os.path.abspath("evaluation")
    env["PYTHONPATH"] = evaluation + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
