"""Save images for a few configs so compiled vs eager runs can be compared: python compile_parity_check.py OUTDIR"""
import os, sys, numpy as np, torch
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
from nanobrag_torch.config import CrystalConfig, DetectorConfig, BeamConfig
from nanobrag_torch.models import Crystal, Detector
from nanobrag_torch.simulator import Simulator
out=sys.argv[1]; os.makedirs(out, exist_ok=True); dev="cuda"
for label,npix,mos,phi in [("a",512,1,1),("b",512,10,1),("c",512,10,10),("d",1024,1,1)]:
    for dt in (torch.float32, torch.float64):
        cc=CrystalConfig(N_cells=(5,5,5),default_F=100.0,mosaic_spread_deg=(0.5 if mos>1 else 0.0),mosaic_domains=mos,mosaic_seed=7,osc_range_deg=(1.0 if phi>1 else 0.0),phi_steps=phi)
        dc=DetectorConfig(distance_mm=100,pixel_size_mm=0.1,spixels=npix,fpixels=npix,oversample=1)
        sim=Simulator(Crystal(cc,device=dev,dtype=dt),Detector(dc,device=dev,dtype=dt),beam_config=BeamConfig(wavelength_A=1.0),device=dev,dtype=dt)
        img1=sim.run().detach().cpu().numpy(); img2=sim.run().detach().cpu().numpy()
        np.save(f"{out}/{label}_{str(dt)[6:]}_run1.npy", img1); np.save(f"{out}/{label}_{str(dt)[6:]}_run2.npy", img2)
print("saved to", out)
