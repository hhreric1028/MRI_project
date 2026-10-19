"""Isolated AMP contract tests; no MRI sampling, checkpoint update, or training run."""
import unittest
import torch
from amp_resume_trainer import amp_update,all_finite,tensors_in

class AmpTests(unittest.TestCase):
    def test_transient_inf_and_nan_skip_then_recover(self):
        for bad in (float('inf'),float('nan')):
            net=torch.nn.Linear(2,1)
            optimizer=torch.optim.SGD(net.parameters(),lr=.01,momentum=.99,nesterov=True)
            scaler=torch.amp.GradScaler('cpu',init_scale=1048576)
            x=torch.ones(1,2)
            amp_update(net(x).square().mean(),net,optimizer,scaler)
            optimizer.zero_grad(set_to_none=True)
            before={k:v.detach().clone() for k,v in net.state_dict().items()}
            momentum=[t.clone() for t in tensors_in(optimizer.state)]
            hook=net.weight.register_hook(lambda grad:torch.full_like(grad,bad))
            event=amp_update(net(x).square().mean(),net,optimizer,scaler)
            hook.remove()
            self.assertTrue(event['optimizer_step_skipped']);self.assertFalse(event['gradient_finite'])
            self.assertEqual(event['scale_after'],event['scale_before']*.5)
            for k,v in net.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
            for a,b in zip(momentum,tensors_in(optimizer.state)):self.assertTrue(torch.equal(a,b))
            optimizer.zero_grad(set_to_none=True)
            recovered=amp_update(net(x).square().mean(),net,optimizer,scaler)
            self.assertFalse(recovered['optimizer_step_skipped']);self.assertTrue(recovered['gradient_finite'])
            self.assertTrue(all_finite(net.parameters()));self.assertTrue(all_finite(tensors_in(optimizer.state)))

    def test_finite_update_matches_original_clipping(self):
        net=torch.nn.Linear(2,1);reference=torch.nn.Linear(2,1);reference.load_state_dict(net.state_dict())
        a=torch.optim.SGD(net.parameters(),lr=.01,momentum=.99,nesterov=True)
        b=torch.optim.SGD(reference.parameters(),lr=.01,momentum=.99,nesterov=True)
        sa=torch.amp.GradScaler('cpu');sb=torch.amp.GradScaler('cpu')
        x=torch.full((2,2),10.)
        amp_update(net(x).square().mean(),net,a,sa)
        sb.scale(reference(x).square().mean()).backward();sb.unscale_(b)
        torch.nn.utils.clip_grad_norm_(reference.parameters(),12);sb.step(b);sb.update()
        for u,v in zip(net.parameters(),reference.parameters()):self.assertTrue(torch.equal(u,v))

    def test_non_amp_nonfinite_stops(self):
        net=torch.nn.Linear(1,1);optimizer=torch.optim.SGD(net.parameters(),lr=.01)
        hook=net.weight.register_hook(lambda grad:torch.full_like(grad,float('inf')))
        with self.assertRaises(FloatingPointError):amp_update(net(torch.ones(1,1)).sum(),net,optimizer,None)
        hook.remove()

    def test_poly_position(self):
        from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.ones(1))],lr=.01)
        scheduler=PolyLRScheduler(optimizer,.01,1000,.9)
        scheduler.step(33);before=optimizer.param_groups[0]['lr']
        scheduler.step(34);after=optimizer.param_groups[0]['lr']
        self.assertAlmostEqual(before,.009702503915514164,places=14)
        self.assertAlmostEqual(after,.01*(1-34/1000)**.9,places=14)
        self.assertLess(after,before)

if __name__=='__main__':unittest.main()
