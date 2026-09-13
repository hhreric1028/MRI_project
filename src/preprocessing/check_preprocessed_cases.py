"""Read-only audit of official outputs, with five in-memory transformation replays."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/.matplotlib'))
os.environ['nnUNet_raw'] = str(ROOT / 'datasets/ISLES2022/nnUNet_raw')
os.environ['nnUNet_preprocessed'] = str(ROOT / 'datasets/ISLES2022/processed')
os.environ['nnUNet_results'] = str(ROOT / 'outputs/models')
import json
import random
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape


def plot_case(output, case, config, original, original_seg, image, seg, prop, spacing, original_spacing):
    peak = int(np.argmax(np.count_nonzero(seg[0] == 1, axis=(1, 2))))
    if not np.any(seg == 1):
        peak = image.shape[1] // 2
    zs = [max(0, peak - 1), peak, min(image.shape[1] - 1, peak + 1)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    for col, z in enumerate(zs):
        # Pixel-center mapping to the nearest original plane; no data resampling for display.
        oz = int(round((z + 0.5) * prop['shape_after_cropping_and_before_resampling'][0] / image.shape[1] - 0.5 + prop['bbox_used_for_cropping'][0][0]))
        oz = int(np.clip(oz, 0, original.shape[1] - 1))
        for row, (im, gt, index, sp) in enumerate(((original, original_seg, oz, original_spacing), (image, seg, z, spacing))):
            plane = im[0, index]
            finite = plane[np.isfinite(plane) & (plane != 0)]
            lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0, 1)
            axes[row, col].imshow(plane, cmap='gray', origin='lower', vmin=lo, vmax=hi, aspect=sp[1] / sp[2])
            mask = gt[0, index] == 1
            if mask.any():
                axes[row, col].contour(mask, levels=[0.5], colors=['red'], linewidths=0.7)
            axes[row, col].set_title(f'{"Original" if row == 0 else "Preprocessed"} z={index}')
            axes[row, col].axis('off')
    fig.suptitle(f'{case} | {config}\nLesion peak and adjacent slices; metadata-matched original planes; native array axes')
    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.03, top=0.85, wspace=0.12, hspace=0.24)
    fig.savefig(output / f'{case}_{config}.png', dpi=130)
    plt.close(fig)


def main():
    folders = sorted((ROOT / 'datasets/ISLES2022/nnUNet_raw').glob('Dataset*'))
    if len(folders) != 1:
        raise RuntimeError('Expected one unambiguous dataset')
    raw = folders[0]
    processed = ROOT / 'datasets/ISLES2022/processed' / raw.name
    manager = PlansManager(str(processed / 'nnUNetPlans.json'))
    dataset_json = json.loads((raw / 'dataset.json').read_text())
    identifiers = sorted(p.name[:-7] for p in (raw / 'labelsTr').glob('*.nii.gz'))
    samples = sorted(random.Random(2022).sample(identifiers, 5))
    mapping = {row['nnunet_case']: row['original_case'] for row in csv.DictReader((ROOT / 'datasets/ISLES2022/documentation/case_mapping.csv').open())}
    inspection = json.loads((ROOT / 'outputs/metrics/data/isles2022_inspection.json').read_text())
    report = dict(dataset=raw.name, expected=len(identifiers), seed=2022, samples=samples, configurations={}, failures=[])
    output = ROOT / 'outputs/figures/preprocessing'
    output.mkdir(parents=True, exist_ok=True)
    for name in manager.available_configurations:
        cfg = manager.get_configuration(name)
        folder = processed / cfg.data_identifier
        loader = infer_dataset_class(str(folder))(str(folder))
        missing = sorted(set(identifiers) - set(loader.identifiers))
        extra = sorted(set(loader.identifiers) - set(identifiers))
        result = dict(cases={}, missing=missing, extra=extra, labels=[], samples={}, failures=[])
        values = set()
        for case in identifiers:
            if case in missing:
                continue
            try:
                data, seg, _, prop = loader.load_case(case)
                data, seg = data[:], seg[:]
                spacing = [prop['spacing'][i] for i in manager.transpose_forward]
                target = list(cfg.spacing)
                if len(target) == 2:
                    target = [spacing[0]] + target
                expected_shape = compute_new_shape(prop['shape_after_cropping_and_before_resampling'], spacing, target).tolist()
                labels = np.unique(seg).tolist()
                values.update(labels)
                entry = dict(shape=list(data.shape), segmentation_shape=list(seg.shape), target_spacing=target,
                             expected_shape=expected_shape, unique_labels=labels,
                             finite=bool(np.isfinite(data).all()), lesion_voxels=int(np.count_nonzero(seg == 1)))
                if data.shape != seg.shape or list(data.shape[1:]) != expected_shape or not entry['finite'] or not set(labels).issubset({-1, 0, 1}):
                    raise RuntimeError(f'Invalid geometry, values or labels: {entry}')
                result['cases'][case] = entry
                if case in samples:
                    image_file = str(raw / 'imagesTr' / f'{case}_0000.nii.gz')
                    seg_file = str(raw / 'labelsTr' / f'{case}.nii.gz')
                    # Official transform repeated in RAM only: never saves or changes preprocessed files.
                    replay_data, replay_seg, _ = cfg.preprocessor_class(verbose=False).run_case([image_file], seg_file, manager, cfg, dataset_json)
                    check = dict(image_transform_matches=bool(np.allclose(data, replay_data, rtol=1e-6, atol=1e-6)),
                                 segmentation_transform_matches=bool(np.array_equal(seg, replay_seg)),
                                 max_image_difference=float(np.max(np.abs(data - replay_data))))
                    if not check['image_transform_matches'] or not check['segmentation_transform_matches']:
                        raise RuntimeError(f'Official transform replay mismatch: {check}')
                    original, original_properties = manager.image_reader_writer_class().read_images([image_file])
                    original_seg, seg_properties = manager.image_reader_writer_class().read_seg(seg_file)
                    for field in ('origin', 'direction', 'spacing'):
                        if not np.allclose(original_properties['sitk_stuff'][field], seg_properties['sitk_stuff'][field], rtol=0, atol=1e-5):
                            raise RuntimeError(f'Original image/GT {field} mismatch')
                    axes = [0] + [i + 1 for i in manager.transpose_forward]
                    original, original_seg = original.transpose(axes), original_seg.transpose(axes)
                    orig_info = inspection['cases'][mapping[case]]['modalities']
                    check.update(original_case=mapping[case], original_shape_xyz=orig_info['dwi']['shape'],
                                 original_spacing_xyz=orig_info['dwi']['spacing'], original_gt_shape_xyz=orig_info['msk']['shape'],
                                 original_gt_spacing_xyz=orig_info['msk']['spacing'], original_orientation=orig_info['dwi']['orientation'],
                                 preprocessed_shape=list(data.shape[1:]), target_spacing=target,
                                 bbox=prop['bbox_used_for_cropping'])
                    result['samples'][case] = check
                    plot_case(output, case, name, original, original_seg, data, seg, prop, target, spacing)
            except Exception as exc:
                result['failures'].append(dict(case=case, error=str(exc)))
        result['labels'] = sorted(values)
        result['successfully_preprocessed'] = len(result['cases']) - len([f for f in result['failures'] if f['case'] in result['cases']])
        result['median_spatial_shape'] = np.median([c['shape'][1:] for c in result['cases'].values()], axis=0).tolist()
        report['configurations'][name] = result
        report['failures'].extend(dict(configuration=name, **failure) for failure in result['failures'])
        print(name, 'readable', len(result['cases']), 'missing', missing, 'failures', result['failures'], 'labels', sorted(values), flush=True)
    (ROOT / 'outputs/metrics/preprocessing/preprocessing_qa.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if report['failures'] or any(c['missing'] or c['extra'] for c in report['configurations'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
