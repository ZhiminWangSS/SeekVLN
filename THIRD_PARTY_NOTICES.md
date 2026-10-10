# Third-Party Notices

This inference preparation copy retains upstream source headers and module
paths. It is not yet a public release; the repository-level license must be
confirmed before publication.

- `llava/` includes code derived from [LLaVA](https://github.com/haotian-liu/LLaVA),
  [VILA](https://github.com/NVlabs/VILA), and [NaVILA](https://github.com/AnjieCheng/NaVILA).
  Apache-2.0 notices in individual files are preserved; the license text is
  included at `licenses/NaVILA-Apache-2.0.txt`.
- The projector compatibility implementation supports AuxThink-format weights.
- `evaluation/` includes VLN-CE extensions under the
  [VLN-CE MIT license](https://github.com/jacobkrantz/VLN-CE/blob/master/LICENSE),
  included at `licenses/VLN-CE-MIT.txt`.
- Habitat, Matterport3D scenes, checkpoints, and dataset assets have separate
  terms. A code license does not grant redistribution rights to these assets.

Files under `llava/train/` are retained only where imported by the existing
model implementation. No training entry points, recipes, VERL package, or
TRL training engines are included.
