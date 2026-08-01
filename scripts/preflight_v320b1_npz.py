from pathlib import Path
import numpy as np
r=Path(r'results\cic_iot_diad_untargeted_exact_qualification_v320a3\runs')
bs=sorted(r.glob('*/*/plain_fedavg/attack_manifest'))
assert len(bs)==20,f'Expected 20 branches, found {len(bs)}'
bad=[]
for d in bs:
 with np.load(d/'poisoned_indices.npz',allow_pickle=False) as i,np.load(d/'poisoned_labels.npz',allow_pickle=False) as l:
  for c in range(20):
   g=f'client_{c:03d}_global_indices';p=f'client_{c:03d}_local_positions';k=f'client_{c:03d}'
   if g not in i.files or p not in i.files or k not in l.files: bad.append((str(d.relative_to(r)),c,'missing key'));continue
   a=np.asarray(i[g]);b=np.asarray(i[p]);z=np.asarray(l[k])
   if a.ndim!=1 or b.ndim!=1 or z.ndim!=1 or len(a)!=len(b) or len(a)!=len(z) or np.any(a<0) or np.any(b<0): bad.append((str(d.relative_to(r)),c,'invalid arrays'))
assert not bad,f'NPZ PREFLIGHT FAILED: {bad[:20]}'
print(f'NPZ PREFLIGHT PASSED: {len(bs)} branches x 20 clients')
