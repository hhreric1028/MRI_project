"""Small synthetic checks for dangerous inventory/geometry/label edge cases."""
import sys
import tempfile
import unittest
from pathlib import Path
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/data'))
from inspect_isles2022 import discover, inspect, geometry


class DataChecks(unittest.TestCase):
    def test_geometry_rejects_translation_and_spacing(self):
        info = dict(shape=[3, 4, 5], spacing=[1, 1, 1], affine=np.eye(4).tolist(), orientation=['R', 'A', 'S'])
        shifted = dict(info, affine=[[1, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.assertFalse(geometry(info, shifted)['affine_compatible'])
        self.assertFalse(geometry(info, dict(info, spacing=[1, 1, 2]))['spacing_equal'])
        self.assertTrue(all(geometry(info, info).values()))

    def test_labels_are_observed_not_binarized(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as directory:
            path = Path(directory) / 'mask.nii.gz'
            data = np.zeros((3, 4, 5), dtype=np.float32)
            data[1, 1, 1] = 2.5
            nib.save(nib.Nifti1Image(data, np.eye(4)), path)
            result = inspect(path, True)
            self.assertEqual(result['unique_values'], [0.0, 2.5])
            self.assertFalse(result['integer_labels'])
            self.assertNotIn('lesion_voxel_count', result)

    def test_duplicates_and_gt_only_cases_are_visible(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as directory:
            raw = Path(directory)
            for sub in ('a', 'b'):
                (raw / sub).mkdir()
                (raw / sub / 'sub-strokecase0001_ses-0001_dwi.nii.gz').touch()
            (raw / 'sub-strokecase0002_ses-0001_msk.nii.gz').touch()
            cases = discover(raw)
            self.assertEqual(len(cases), 2)
            self.assertEqual(len(cases['sub-strokecase0001_ses-0001']['dwi']), 2)
            self.assertEqual(set(cases['sub-strokecase0002_ses-0001']), {'msk'})

    def test_nifti_scaling_artifact_is_preserved_and_audited(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as directory:
            path = Path(directory) / 'mask.nii.gz'
            data = np.full((3, 4, 5), -32768, dtype=np.int16)
            data[1, 1, 1] = 32767
            img = nib.Nifti1Image(data, np.eye(4))
            img.header.set_slope_inter(1.5259021893143654e-05, 0.5000076293945312)
            nib.save(img, path)
            before = path.read_bytes()
            result = inspect(path, True)
            self.assertFalse(result['integer_labels'])
            self.assertTrue(result['near_integer_labels'])
            self.assertEqual(result['nnunet_reader_unique_values'], [0.0, 1.0])
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
