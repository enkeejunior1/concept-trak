# AbC Experiment

This directory hosts the self-contained `abc` reproduction pipeline for
`concept-trak`.

## Layout

- `preliminary/prepare_data.py`: copy or symlink the AbC benchmark artifacts into the local experiment layout.
- `utils.py`: copied benchmark utilities and dataset helpers.
- `train_loss.py`, `train_grad.py`: train-side latent attribution features.
- `task_loss.py`, `task_grad.py`: exemplar/task-side features.
- `baseline_local_grad.py`, `baseline_global_grad.py`: baseline concept-side gradients.
- `test_global_grad.py`, `test_local_grad.py`, `test_local_alt_grad.py`, `test_local_seed_grad.py`, `test_local_ti_grad.py`: concept-slider variants.
- `ti.py`: textual inversion stage used by the TI-based local concept path.
- `qual.py`: single-file qualitative run that generates one query image from a fixed `xT`, then computes seed-based concept attribution, recall metrics, and top-K visualization.
- `influence.py`: influence score computation.
- `eval.py`: recall metric computation and top-K visualization from saved influence results.

## Expected Local Layout

After running the preparation script, the experiment directory is expected to
contain:

- `data/laion_latents.npy`
- `data/laion_text_embeddings.npy`
- `data/laion_subset/`
- `data/json/`
- `configs/all_tasks.json`
- task-specific benchmark assets referenced from `all_tasks.json`, including model and synthesized-image paths

## Notes

- The implementation is copied from the original private `abc` code and patched only enough to use the local `experiments/abc` layout.
- The code still depends on external AbC benchmark assets and a Stable Diffusion v1.4 checkpoint, but it no longer depends on `concept-trak-old/abc` at runtime.
