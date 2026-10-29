"""Check saved inference on released audit data, without loading competition inputs."""
import argparse,json
import numpy as np
import torch
from simulation import BASE
from select_and_audit import Ensemble

def main():
    p=argparse.ArgumentParser();p.add_argument('--tree',default='final');p.add_argument('--neural',default='neural_final');args=p.parse_args()
    torch.set_num_threads(5)
    x=np.load(BASE/'reports/audit_examples.npz')['db'][:32]
    model=Ensemble(args.tree,args.neural,'cpu')
    pred,ev,_=model.predict_components(x)
    assert np.isfinite(pred).all() and np.isfinite(ev).all()
    assert ((pred>=0)&(pred<=1)).all() and ((ev>=0)&(ev<=1)).all()
    reverse,reverse_ev,_=model.predict_components(x[::-1].copy())
    np.testing.assert_allclose(pred,reverse[::-1],atol=2e-6,rtol=2e-5)
    np.testing.assert_allclose(ev,reverse_ev[::-1],atol=1e-12)
    independent=np.stack([model.predict_components(x[i:i+1])[0][0] for i in [0,5,15,31]])
    np.testing.assert_allclose(pred[[0,5,15,31]],independent,atol=2e-6,rtol=2e-5)
    record={'finite_and_bounded':True,'row_order_invariance':True,'per_window_independence':True,'max_row_order_difference':float(np.max(abs(pred-reverse[::-1]))),'max_single_window_difference':float(np.max(abs(pred[[0,5,15,31]]-independent))),'test_acoustic_values_read':False}
    (BASE/'reports/inference_tests.json').write_text(json.dumps(record,indent=2))
    print(record)

if __name__=='__main__':main()
