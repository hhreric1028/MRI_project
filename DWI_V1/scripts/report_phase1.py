"""Generate measured phase-one report and a full project tree."""
import importlib.metadata
import json
import platform
from collections import Counter
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / 'outputs/metrics/data/isles2022_inspection.json').read_text())
    summary = report['summary']
    dataset_path = ROOT / 'datasets/ISLES2022/nnUNet_raw/Dataset001_ISLES2022_DWI'
    dataset = (dataset_path / 'dataset.json').read_text()
    versions = dict(python=platform.python_version(), pytorch=torch.__version__, cuda_runtime=torch.version.cuda,
                    nnunetv2=importlib.metadata.version('nnunetv2'), gpu=torch.cuda.get_device_name(0), conda_environment='MRI_DWI')
    (ROOT / 'docs/environment/runtime-versions.json').write_text(json.dumps(versions, indent=2))
    lines = ['# Phase 1 completion report', '', '## Environment', '', '```json', json.dumps(versions, indent=2), '```', '',
             '## Source and files', '', '[Official ISLES2022 v2.3.1 release](https://zenodo.org/records/7960856)', '',
             f'Raw: `{ROOT / "datasets/ISLES2022/raw"}`', '',
             'Official archive MD5 and extracted file hashes are recorded in documentation/.', '',
             '## Measured completeness', '', '```json', json.dumps(summary, indent=2), '```', '',
             f"Observed GT values: `{report['labels']['observed_unique_values']}`.", '',
             f"Official SimpleITKIO values: `{report['labels']['nnunet_reader_unique_values']}`. No mask rewriting or label remapping.", '',
             f"Empty masks: {sum(c['modalities']['msk']['empty_mask'] for c in report['cases'].values())}.", '',
             'Empty-mask case IDs (all retained): ' + ', '.join(k for k, c in report['cases'].items() if c['modalities']['msk']['empty_mask']), '',
             'NIfTI scaling artifact case IDs (native float64 foreground is near 1; official reader gives exactly 1; all files copied unchanged): ' + ', '.join(k for k, c in report['cases'].items() if not c['modalities']['msk']['integer_labels']), '',
             '## Five random samples (seed 2022)', '',
             '| Case | Modality | Shape | Spacing (mm) | Orientation |', '|---|---|---|---|---|']
    for case_id in report['sample_cases']:
        for modality, info in report['cases'][case_id]['modalities'].items():
            lines.append(f"| {case_id} | {modality.upper()} | {info['shape']} | {[round(x, 6) for x in info['spacing']]} | {''.join(info['orientation'])} |")
    lines += ['', '## Geometry differences', '', 'Tolerances: spacing absolute 1e-5; affine absolute 1e-3 (no relative tolerance).', '',
              f"Differences by comparison modality: {dict(Counter(i['modality'] for i in report['geometry_issues']))}.", '',
              'Full case-by-case matrices, datatype, labels and counts are in outputs/metrics/data/isles2022_inspection.json and .csv.', '',
              '| Case | DWI compared with | Differences |', '|---|---|---|']
    for issue in report['geometry_issues']:
        failed = ', '.join(k for k, v in issue.items() if v is False)
        lines.append(f"| {issue['case_id']} | {issue['modality']} | {failed} |")
    lines += ['', 'Grid agreement does not by itself prove anatomical alignment. See visual-review.md for the five inspected QC images. No resampling was performed.', '',
              '## nnUNet dataset', '', f'`{dataset_path}`', '', '```json', dataset, '```', '',
              '## Official integrity check', '', '```text', (ROOT / 'outputs/metrics/data/nnunet_integrity.log').read_text(encoding='utf-8-sig'), '```', '',
              'The official planning CLI help was checked, and the same official integrity routine was run directly. No fingerprint extraction, planning, preprocessing or training was started.', '',
              '## Directory tree', '', 'Full tree: [historical directory-tree.txt](reference/directory-tree.txt). Raw and converted files are all included.']
    (ROOT / 'docs/phase1_report.md').write_text('\n'.join(lines), encoding='utf-8')
    def tree(folder, prefix=''):
        children = sorted((p for p in folder.iterdir() if p.name != '__pycache__'), key=lambda p: (not p.is_dir(), p.name))
        result = []
        for index, child in enumerate(children):
            last = index == len(children) - 1
            result.append(prefix + ('└── ' if last else '├── ') + child.name + ('/' if child.is_dir() else ''))
            if child.is_dir():
                result.extend(tree(child, prefix + ('    ' if last else '│   ')))
        return result
    tree_path = ROOT / 'docs/reference/directory-tree.txt'
    tree_path.touch(exist_ok=True)
    tree_path.write_text(str(ROOT) + '\n' + '\n'.join(tree(ROOT)), encoding='utf-8')


if __name__ == '__main__':
    main()
