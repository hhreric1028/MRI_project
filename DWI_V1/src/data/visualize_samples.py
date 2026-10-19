"""Native-grid QC: DWI/ADC with GT where geometry matches; native FLAIR separately."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[2] / 'outputs/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from inspect_isles2022 import ROOT, geometry


def main():
    report = json.loads((ROOT / 'outputs/metrics/data/isles2022_inspection.json').read_text())
    output = ROOT / 'outputs/figures/data'
    output.mkdir(parents=True, exist_ok=True)
    for case_id in report['sample_cases']:
        infos = report['cases'][case_id]['modalities']
        mask = np.asanyarray(nib.load(ROOT / infos['msk']['path']).dataobj)
        z = int(np.argmax(np.count_nonzero(mask, axis=(0, 1))))
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        for ax, modality in zip(axes, ('dwi', 'adc', 'flair', 'msk')):
            info = infos[modality]
            data = np.asanyarray(nib.load(ROOT / info['path']).dataobj)
            compatible = all(geometry(infos['msk'], info).values())
            slice_z = z if compatible else data.shape[2] // 2
            plane = data[:, :, slice_z].T
            nonzero = plane[np.isfinite(plane) & (plane != 0)]
            lo, hi = np.percentile(nonzero, [1, 99]) if nonzero.size else (0, 1)
            ax.imshow(plane, origin='lower', cmap='gray', vmin=lo if modality != 'msk' else 0,
                      vmax=hi if modality != 'msk' else 1, aspect=info['spacing'][1] / info['spacing'][0])
            if compatible and modality != 'msk' and mask[:, :, z].any():
                ax.contour(mask[:, :, z].T, levels=[0.5], colors=['red'], linewidths=0.6)
            ax.set_title(f'{modality.upper()} z={slice_z}\n' + ('same grid as GT' if compatible else 'native grid; no overlay'))
            ax.axis('off')
        fig.suptitle(case_id + '\nNative voxel axes; no reorientation, registration or resampling')
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        fig.savefig(output / f'{case_id}_qc.png', dpi=130)
        plt.close(fig)


if __name__ == '__main__':
    main()
