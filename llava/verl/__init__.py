"""SeekVLN extensions for VERL.

The package deliberately has no eager VERL imports.  Pure algorithm tests can
therefore run in the existing NaVILA environment, while ``llava.verl.register``
is imported by VERL workers during a distributed training job.
"""

__all__ = []
