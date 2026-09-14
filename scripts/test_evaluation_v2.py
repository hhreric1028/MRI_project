import unittest
import numpy as np
from evaluate_experiment import metrics, summarize

class EvaluationTests(unittest.TestCase):
    def test_empty_and_distance(self):
        a=np.zeros((7,7,7),bool);b=a.copy();a[2,2,2]=1;b[3,2,2]=1
        self.assertEqual(metrics(a,b,np.diag([2,3,4,1]),.024)['hd95_mm'],2)
        self.assertEqual(metrics(a,a,np.eye(4),.001)['dice'],1)
        z=np.zeros_like(a)
        self.assertIsNone(metrics(z,z,np.eye(4),.001)['dice'])
        self.assertEqual(metrics(a,z,np.eye(4),.001)['hd95_status'],'infinite_single_empty')
        self.assertIsNone(metrics(z,a,np.eye(4),.001)['absolute_percentage_error'])
    def test_merge_and_split(self):
        g=np.zeros((9,9,9),bool);g[2,4,4]=1;g[6,4,4]=1
        p=g.copy();p[2:7,4,4]=1
        m=metrics(g,p,np.eye(4),.001)
        self.assertEqual((m['lesion_tp'],m['lesion_fn'],m['lesion_fp']),(1,1,0))
        m=metrics(p,g,np.eye(4),.001)
        self.assertEqual((m['lesion_tp'],m['lesion_fn'],m['lesion_fp']),(1,0,1))
    def test_micro_aggregation(self):
        z=np.zeros((3,3,3),bool);a=z.copy();a[1,1,1]=1
        s=summarize([metrics(a,a,np.eye(4),.001),metrics(a,z,np.eye(4),.001)])
        self.assertEqual(s['lesion_recall'],.5)
        self.assertEqual(s['hd95_single_empty'],1)
        self.assertEqual(s['severe_under_count'],1)

if __name__=='__main__':unittest.main()
