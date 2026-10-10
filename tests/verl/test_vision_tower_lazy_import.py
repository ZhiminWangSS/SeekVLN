import subprocess
import sys


def test_vision_tower_builder_does_not_import_optional_intern_backend():
    code = """
import sys
import llava.model.multimodal_encoder.builder
assert 'llava.model.multimodal_encoder.intern_encoder' not in sys.modules
assert 'llava.model.multimodal_encoder.intern.flash_attention' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
