from __future__ import annotations
import json, os, shutil, time
from pathlib import Path

if os.environ.get("LFIGHTER_TASK65_GUARD") == "1":
    import numpy as np
    import torch
    import federated_iot_v26 as fed

    _ALLOWED = ("X_train","y_train","X_val","y_val")
    _TEST = ("X_test_natural","y_test_natural","X_test_diagnostic","y_test_diagnostic")
    _CAPTURE_NAMES = {"continuation_last_round_model.pt","v320b1_last_round_model.pt"}
    _LOG = os.environ.get("LFIGHTER_TASK65_GUARD_LOG","")

    def _log(event):
        if not _LOG:
            return
        p=Path(_LOG); p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a",encoding="utf-8") as h:
            h.write(json.dumps({"time_unix":time.time(),**event},sort_keys=True)+"\n")

    def guarded_load_protocol_arrays(path):
        path=Path(path).expanduser().resolve()
        with np.load(path,allow_pickle=False) as data:
            names=set(data.files)
            missing=[k for k in (*_ALLOWED,*_TEST) if k not in names]
            if missing:
                raise KeyError(f"Task65 expected protocol array names missing: {missing}")
            arrays={k:data[k] for k in _ALLOWED}
        _log({"event":"guarded_load_protocol_arrays","materialized_keys":list(_ALLOWED),
              "test_array_names_verified_only":list(_TEST),"test_arrays_materialized":False})
        return arrays

    guarded_load_protocol_arrays.__name__="task65_guarded_load_protocol_arrays"
    guarded_load_protocol_arrays.__module__=__name__
    fed.load_protocol_arrays=guarded_load_protocol_arrays

    _orig_save=torch.save
    def guarded_torch_save(obj,f,*args,**kwargs):
        result=_orig_save(obj,f,*args,**kwargs)
        try:
            path=Path(f)
            if path.name in _CAPTURE_NAMES and isinstance(obj,dict):
                r=int(obj.get("global_round",-1))
                if r in (5,6,7,8):
                    d=path.parent/"task65_round_checkpoints"
                    d.mkdir(parents=True,exist_ok=True)
                    dst=d/f"global_round_{r:02d}_model.pt"
                    shutil.copy2(path,dst)
                    _log({"event":"round_checkpoint_captured","source_name":path.name,
                          "global_round":r,"capture":str(dst.resolve())})
        except Exception as exc:
            _log({"event":"checkpoint_capture_error","error":repr(exc)})
            raise
        return result
    torch.save=guarded_torch_save
    _log({"event":"task65_guard_loaded","test_arrays_materialized":False,
          "capture_names":sorted(_CAPTURE_NAMES)})
