"""Copy verified DWI/GT pairs into nnUNet format; never alter raw files."""
import csv
import hashlib
import json
import shutil
from pathlib import Path
from inspect_isles2022 import ROOT, discover, inspect, geometry


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    raw = ROOT / 'datasets/ISLES2022/raw'
    destination = ROOT / 'datasets/ISLES2022/nnUNet_raw/Dataset001_ISLES2022_DWI'
    cases = discover(raw)
    if len(cases) != 250:
        raise RuntimeError(f'Expected official 250 cases, found {len(cases)}; no cases excluded automatically.')
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    verified = []
    values = set()
    for case_id, paths in cases.items():
        if any(len(paths.get(m, [])) != 1 for m in ('dwi', 'adc', 'flair', 'msk')):
            raise RuntimeError(f'Incomplete or duplicate case: {case_id}')
        dwi, mask = paths['dwi'][0], paths['msk'][0]
        a, b = inspect(dwi), inspect(mask, True)
        if not all(geometry(a, b).values()) or not a['finite'] or not b['finite'] or not b['near_integer_labels'] or len(a['shape']) != 3:
            raise RuntimeError(f'DWI/GT failed preflight: {case_id}')
        values.update(b['nnunet_reader_unique_values'])
        verified.append((case_id, dwi, mask))
    if sorted(values) != [0, 1]:
        raise RuntimeError(f'Observed labels {sorted(values)} require explicit semantic review; no automatic remapping.')
    background, lesion = map(int, sorted(values))
    for name in ('imagesTr', 'labelsTr'):
        (destination / name).mkdir(parents=True)
    rows = []
    for index, (original, dwi, mask) in enumerate(verified, 1):
        case = f'case_{index:04d}'
        row = dict(original_case=original, nnunet_case=case)
        for modality, source, target in (('dwi', dwi, destination / 'imagesTr' / f'{case}_0000.nii.gz'), ('gt', mask, destination / 'labelsTr' / f'{case}.nii.gz')):
            source_hash = digest(source)
            shutil.copy2(source, target)
            if digest(target) != source_hash:
                raise RuntimeError(f'Copy checksum mismatch: {target}')
            row.update({f'{modality}_source': str(source.relative_to(ROOT)), f'{modality}_sha256': source_hash})
        rows.append(row)
    dataset = dict(channel_names={'0': 'DWI'}, labels={'background': background, 'lesion': lesion}, numTraining=len(rows), file_ending='.nii.gz', overwrite_image_reader_writer='SimpleITKIO')
    (destination / 'dataset.json').write_text(json.dumps(dataset, indent=2), encoding='utf-8')
    with (ROOT / 'datasets/ISLES2022/documentation/case_mapping.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dataset, indent=2))
    print(destination)


if __name__ == '__main__':
    main()
