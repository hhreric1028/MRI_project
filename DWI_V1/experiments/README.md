# DWI_V1 experiments

Each named directory is a stable experiment record. Commit lightweight
reproducibility evidence such as `experiment_config.json`, completion markers,
CSV/JSON metric summaries and Markdown reports. Keep checkpoints, nnUNet
`results/`, predictions, preprocessing arrays and logs local; the repository
`.gitignore` excludes them.

Do not rename completed experiment directories without also updating the
reports and scripts that reference their identifiers.

