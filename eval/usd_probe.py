from pxr import Usd, UsdGeom, Gf
import math
S = Usd.Stage.Open("/data/wam/arena/trossen_ai_isaac/assets/robots/mobile_ai/mobile_ai.usd")
print("=== Camera prims ===")
for p in S.Traverse():
    if p.GetTypeName() == "Camera":
        print("CAM:", p.GetPath())
print("=== camera-relevant Xforms (path contains cam/camera/d405) ===")
def local_quat(prim):
    x = UsdGeom.Xformable(prim)
    m = x.GetLocalTransformation()  # Gf.Matrix4d
    r = m.ExtractRotationQuat()  # Gf.Quatd, GetReal + GetImaginary
    im = r.GetImaginary()
    return (r.GetReal(), im[0], im[1], im[2]), m.ExtractTranslation()
for p in S.Traverse():
    nm = p.GetName().lower()
    if any(k in nm for k in ("cam_high","camera_link","camera_mount","d405","optical")) and UsdGeom.Xformable(p):
        try:
            q,t = local_quat(p)
            print(f"{str(p.GetPath()):60s} type={p.GetTypeName():10s} localT=({t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f}) localQ_wxyz=({q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f},{q[3]:+.3f})")
        except Exception as e:
            print(p.GetPath(), "err", e)
