"""Read-only ISLES2022 inventory, geometry and observed-label audit. No resampling."""
import argparse
import csv
import json
import random
import re
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODALITIES = ('dwi', 'adc', 'flair', 'msk')


def discover(raw):
    cases = {}
    for path in sorted(raw.rglob('*.nii*')):
        if not (path.name.endswith('.nii') or path.name.endswith('.nii.gz')):
            continue
        match = re.fullmatch(r'(sub-strokecase\d+)_ses-(\d+)_(dwi|adc|flair|msk)\.nii(?:\.gz)?', path.name, re.I)
        if match:
            subject, session, modality = match.groups()
            key = f'{subject}_ses-{session}'
            cases.setdefault(key, {}).setdefault(modality.lower(), []).append(path)
    return cases


def inspect(path, mask=False):
    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    result = dict(path=str(path.relative_to(ROOT)), shape=list(img.shape),
                  spacing=[float(x) for x in img.header.get_zooms()],
                  affine=img.affine.tolist(), orientation=list(nib.aff2axcodes(img.affine)),
                  datatype=str(img.get_data_dtype()), finite=bool(np.isfinite(data).all()),
                  spatial_unit=img.header.get_xyzt_units()[0])
    if mask:
        from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
        values, counts = np.unique(data, return_counts=True)
        official_data, _ = SimpleITKIO().read_seg(str(path))
        result.update(unique_values=values.tolist(),
                      label_voxel_counts={str(v): int(n) for v, n in zip(values, counts)},
                      nifti_scaling=dict(slope=float(img.dataobj.slope), intercept=float(img.dataobj.inter)),
                      unscaled_unique_values=np.unique(img.dataobj.get_unscaled()).tolist(),
                      nnunet_reader_unique_values=np.unique(official_data).tolist(),
                      near_integer_labels=bool(np.isfinite(data).all() and np.allclose(data, np.round(data), atol=1e-8, rtol=0)),
                      integer_labels=bool(np.isfinite(data).all() and np.equal(data, np.round(data)).all()))
    return result


def geometry(a, b):
    return dict(shape_equal=a['shape'] == b['shape'],
                spacing_equal=len(a['spacing']) == len(b['spacing']) and bool(np.allclose(a['spacing'], b['spacing'], rtol=0, atol=1e-5)),
                affine_compatible=bool(np.allclose(a['affine'], b['affine'], rtol=0, atol=1e-3)),
                orientation_equal=a['orientation'] == b['orientation'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT / 'datasets/ISLES2022/raw')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/metrics/data/isles2022_inspection.json')
    parser.add_argument('--seed', type=int, default=2022)
    args = parser.parse_args()
    cases = discover(args.raw)
    if not cases:
        raise SystemExit('No ISLES2022 NIfTI cases found; download and extract first.')
    report = dict(seed=args.seed, cases={}, errors=[], geometry_issues=[])
    for case_id, paths in cases.items():
        entry = dict(modalities={}, comparisons={})
        for modality, candidates in paths.items():
            if len(candidates) != 1:
                report['errors'].append(f'{case_id}: duplicate {modality}: {candidates}')
                continue
            try:
                info = inspect(candidates[0], modality == 'msk')
                entry['modalities'][modality] = info
                if not info['finite'] or len(info['shape']) != 3:
                    report['errors'].append(f'{case_id}: nonfinite or non-3D {modality}')
            except Exception as exc:
                report['errors'].append(f'{case_id}: {modality}: {exc}')
        if 'dwi' in entry['modalities']:
            for modality in ('adc', 'flair', 'msk'):
                if modality in entry['modalities']:
                    comparison = geometry(entry['modalities']['dwi'], entry['modalities'][modality])
                    entry['comparisons'][f'dwi_vs_{modality}'] = comparison
                    if not all(comparison.values()):
                        report['geometry_issues'].append(dict(case_id=case_id, modality=modality, **comparison))
        report['cases'][case_id] = entry
    values = sorted(set(v for c in report['cases'].values() for v in c['modalities'].get('msk', {}).get('unique_values', [])))
    # Semantics only confirmed AFTER reading the full observed label set, with source evidence.
    official_values = sorted(set(v for c in report['cases'].values() for v in c['modalities'].get('msk', {}).get('nnunet_reader_unique_values', [])))
    confirmed_binary = official_values == [0.0, 1.0] and all(c['modalities'].get('msk', {}).get('near_integer_labels', False) for c in report['cases'].values())
    report['labels'] = dict(observed_unique_values=values, binary_0_1_confirmed=confirmed_binary,
                            nnunet_reader_unique_values=official_values,
                            scaling_note='Native scaled float64 values retained verbatim. Binary semantics validated through official SimpleITKIO and native near-integer tolerance 1e-8. No files rewritten.',
                            semantic_evidence='Official lesion annotations; author evaluation code casts masks to boolean (nonzero foreground). See documentation/author-eval_utils.py.')
    for case in report['cases'].values():
        mask = case['modalities'].get('msk')
        if mask and confirmed_binary:
            count = sum(n for v, n in mask['label_voxel_counts'].items() if abs(float(v) - 1) <= 1e-8)
            mask.update(lesion_voxel_count=count, empty_mask=count == 0)
    incomplete = [k for k, c in report['cases'].items() if set(c['modalities']) != set(MODALITIES)]
    report['summary'] = dict(total_cases=len(cases), **{f'cases_with_{m}': sum(m in c['modalities'] for c in report['cases'].values()) for m in MODALITIES}, incomplete_cases=incomplete)
    report['sample_cases'] = sorted(random.Random(args.seed).sample(sorted(cases), min(5, len(cases))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    with args.output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case_id', 'modality', 'shape', 'spacing', 'orientation', 'datatype', 'unique_values', 'lesion_voxel_count', 'empty_mask'])
        for case_id, case in report['cases'].items():
            for modality, info in case['modalities'].items():
                writer.writerow([case_id, modality] + [info.get(k, '') for k in ('shape', 'spacing', 'orientation', 'datatype', 'unique_values', 'lesion_voxel_count', 'empty_mask')])
    print(json.dumps({k: report[k] for k in ('summary', 'labels', 'sample_cases', 'errors')}, indent=2))
    print(f"Geometry differences: {len(report['geometry_issues'])}; report: {args.output}")
    if report['errors'] or incomplete:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
