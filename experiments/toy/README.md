# Toy Experiment

This directory contains a self-contained `toy` reproduction pipeline copied from
`toy-rebuttal` and reorganized for the public `concept-trak` layout.

## Layout

- `preliminary/generate_data.py`: build the synthetic classifier and generator datasets.
- `preliminary/train_classifier.py`: train the attribute classifier.
- `preliminary/train_model.py`: train the conditional diffusion generator.
- `generate_samples.py`: generate local test samples for local concept gradients.
- `train_loss.py`: compute per-sample train loss artifacts.
- `train_grad.py`: compute per-sample train gradients.
- `test_local_grad.py`: compute local concept gradients from generated samples.
- `test_global_grad.py`: compute global concept slider gradients.
- `influence.py`: compute influence scores and store ranking artifacts.
- `eval.py`: compute recall and render top-K visualizations from saved influence outputs.

## Notes

- The implementation is intentionally close to `toy-rebuttal` to preserve the original algorithmic behavior.
- The hardcoded W&B login key from the old private code was removed; W&B logging is now optional in the preliminary training scripts.
- Paths default to the local experiment directory so this tree can run independently from the old layout.
