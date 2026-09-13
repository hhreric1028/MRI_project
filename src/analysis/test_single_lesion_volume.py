"""Read one binary ISLES2022 GT mask and report voxel-based lesion volumes."""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mask', type=Path, help='One Ground Truth .nii.gz segmentation')
    args = parser.parse_args()
    path = args.mask.resolve()
    nii = nib.load(path)
    seg = np.asanyarray(nii.dataobj)
    spacing = nii.header.get_zooms()[:3]
    unique_values = np.unique(seg)

    print('================================\nSingle Lesion Volume Test\n================================')
    print(f'\nCase:\n{path.name.split("_ses-")[0]}')
    print(f'\nGround Truth:\n{path}')
    print(f'\nShape:\n{seg.shape}')
    print(f'\nSpacing:\n{tuple(float(s) for s in spacing)} mm')
    print(f'\nUnique labels:\n{unique_values.tolist()}')

    if seg.ndim != 3:
        raise SystemExit('STOP: Expected a 3D segmentation.')
    if not np.isfinite(unique_values).all() or not np.isin(unique_values, [0, 1]).all():
        raise SystemExit('STOP: Unexpected labels; no rounding, remapping or analysis performed.')
    if nii.header.get_xyzt_units()[0] != 'mm':
        raise SystemExit('STOP: NIfTI spatial units are not mm.')
    if not np.isfinite(spacing).all() or np.any(np.asarray(spacing) <= 0):
        raise SystemExit('STOP: Invalid voxel spacing.')

    # ISLES2022 expert lesion GT: nonzero foreground per its supplied documentation.
    # Actual values above are checked before using exact binary equality.
    print('\nLabel definition confirmed: 0 = background; 1 = lesion (ISLES2022 GT).')
    lesion_mask = seg == 1
    total_voxels = int(np.count_nonzero(lesion_mask))
    voxel_volume_mm3 = float(spacing[0]) * float(spacing[1]) * float(spacing[2])
    total_volume_mm3 = total_voxels * voxel_volume_mm3
    components, count = label(lesion_mask, structure=np.ones((3, 3, 3), dtype=bool))
    counts = np.bincount(components.ravel(), minlength=count + 1)[1:]
    volumes_mm3 = [int(n) * voxel_volume_mm3 for n in counts]

    print(f'\nVoxel volume:\n{voxel_volume_mm3:.12g} mm3')
    print(f'\nLesion detected:\n{"YES" if total_voxels else "NO"}')
    print(f'\nTotal lesion voxels:\n{total_voxels}')
    print(f'\nTotal lesion volume:\n{total_volume_mm3:.12g} mm3\n{total_volume_mm3 / 1000:.12g} mL')
    print(f'\nNumber of connected lesions:\n{count}')
    print('\n--------------------------------')
    for index, (n, volume) in enumerate(zip(counts, volumes_mm3), 1):
        print(f'\nLesion {index}\nVoxel count:\n{int(n)}\nVolume:\n{volume:.12g} mm3\n{volume / 1000:.12g} mL')

    assert sum(int(n) for n in counts) == total_voxels
    assert np.isclose(sum(volumes_mm3), total_volume_mm3, rtol=1e-12, atol=1e-9)
    print('\nVerification: component voxel and volume sums match totals.')


if __name__ == '__main__':
    main()
