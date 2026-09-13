"""Call the exact official integrity routine used by plan_and_preprocess only."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['nnUNet_raw'] = str(ROOT / 'datasets/ISLES2022/nnUNet_raw')
os.environ['nnUNet_preprocessed'] = str(ROOT / 'datasets/ISLES2022/processed')
os.environ['nnUNet_results'] = str(ROOT / 'outputs/models')


if __name__ == '__main__':
    from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
    from nnunetv2.experiment_planning.verify_dataset_integrity import verify_dataset_integrity
    name = convert_id_to_dataset_name(1)
    print(f'Official dataset ID resolution: 1 -> {name}', flush=True)
    verify_dataset_integrity(str(Path(os.environ['nnUNet_raw']) / name), num_processes=4)
    print('PASS: official integrity verification only; no fingerprint, planning, preprocessing or training.', flush=True)
