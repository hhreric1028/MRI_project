"""CPU tests: fixed Tversky formula and actual train/validation loader separation."""
import json
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from recall_trainers import ForegroundTverskyCE, DS_WEIGHTS, TRAINERS
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from run_recall_experiment import seed_all, make_trainer

class ProtocolTests(unittest.TestCase):
    def test_paired_bootstrap_identity(self):
        from recall_analysis import bootstrap,load_cases,B0
        a=load_cases(B0);result=bootstrap(a,a,sorted(a),n=100)
        for r in result.values():
            self.assertEqual(r['delta'],0)
            self.assertEqual(r['ci95'],[0,0])

    def test_coverage_geometry(self):
        from recall_sampling import geometric_coverage
        coords=np.argwhere(np.ones((8,8,8)))
        params=dict(center_location_in_pixels=[4,4,4],affine=None,elastic_offsets=None)
        self.assertEqual(geometric_coverage(coords,[0,0,0],[8,8,8],[8,8,8],params),1)
        self.assertEqual(geometric_coverage(coords,[0,0,0],[8,8,8],[4,4,4],params),.125)
        params['affine']=np.eye(3)*2
        self.assertEqual(geometric_coverage(coords,[0,0,0],[8,8,8],[4,4,4],params),1)

    def test_tversky_cases_and_gradients(self):
        loss=ForegroundTverskyCE()
        gt=torch.zeros(2,1,8,8,8);gt[1,:,2:6,2:6,2:6]=1
        scenarios=[(gt[:1],torch.zeros_like(gt[:1])),(gt[:1],torch.ones_like(gt[:1])),
                   (gt[1:],gt[1:]),(gt[1:],torch.zeros_like(gt[1:])),(gt[1:],torch.ones_like(gt[1:]))]
        for target,pred in scenarios:
            logits=torch.cat([1-pred,pred],1).mul(2000).sub(1000).requires_grad_()
            value=loss(logits,target);value.backward()
            self.assertTrue(torch.isfinite(value));self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertAlmostEqual(float(loss.tversky(torch.zeros(1,8),torch.zeros(1,8))),0.)
        g=torch.tensor([[1.,1.,1.,1.,0.,0.,0.,0.]])
        perfect=g.clone();fp=g.clone();fp[0,4]=1;fn=g.clone();fn[0,0]=0
        self.assertGreater(loss.tversky(fp,g),loss.tversky(perfect,g))
        self.assertGreater(loss.tversky(fn,g),loss.tversky(fp,g))
        fn2=fn.clone();fn2[0,1]=0;fp2=fp.clone();fp2[0,5]=1
        self.assertGreater(loss.tversky(fn2,g),loss.tversky(fn,g))
        self.assertGreater(loss.tversky(fp2,g),loss.tversky(fp,g))

    def test_deep_supervision(self):
        loss=ForegroundTverskyCE();wrapped=DeepSupervisionWrapper(loss,DS_WEIGHTS)
        for mode in ('empty','perfect','all_background','all_foreground','random'):
            xs=[];ys=[]
            for size in (16,8,4,2):
                y=torch.zeros(2,1,size,size,size);y[1,:,:size//2]=1
                if mode=='empty':y.zero_()
                if mode in ('empty','all_background'):p=torch.zeros_like(y)
                elif mode=='all_foreground':p=torch.ones_like(y)
                elif mode=='perfect':p=y.clone()
                else:p=torch.rand_like(y)
                x=torch.cat([1-p,p],1).mul(80).sub(40).requires_grad_()
                one=loss(x,y);grad=torch.autograd.grad(one,x,retain_graph=True)[0]
                self.assertTrue(torch.isfinite(one));self.assertTrue(torch.isfinite(grad).all())
                xs.append(x);ys.append(y)
            v=wrapped(xs,ys);v.backward();self.assertTrue(torch.isfinite(v))
            for x in xs[:3]:self.assertTrue(torch.isfinite(x.grad).all())
            self.assertIsNone(xs[-1].grad) # Zero-weight head is intentionally omitted.

    def test_same_seed(self):
        import random
        seed_all();a=(random.random(),np.random.rand(),torch.rand(3))
        seed_all();b=(random.random(),np.random.rand(),torch.rand(3))
        self.assertEqual(a[:2],b[:2]);self.assertTrue(torch.equal(a[2],b[2]))

    def test_actual_loader_ratios(self):
        with tempfile.TemporaryDirectory(prefix='recall_protocol_') as tmp:
            os.environ['nnUNet_results']=tmp
            for name,ratio in [('B0-R2022',.33),('S1a',.50),('L2a',.33)]:
                trainer=make_trainer(name,'cpu');tr,va=trainer.raw_loaders()
                self.assertEqual(tr.oversample_foreground_percent,ratio)
                self.assertEqual(va.oversample_foreground_percent,.33)
                self.assertEqual(sum(tr.get_do_oversample(i) for i in range(8)),4 if ratio==.5 else 3)
                self.assertEqual(sum(va.get_do_oversample(i) for i in range(8)),3)
                self.assertEqual(set(tr._data.identifiers),set(trainer.do_split()[0]))
                self.assertEqual(set(va._data.identifiers),set(trainer.do_split()[1]))

if __name__=='__main__':unittest.main()
