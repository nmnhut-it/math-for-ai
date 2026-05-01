# Lab 2 CNN — entry point: chay full pipeline.
# Tat ca .py thuc te nam trong src/, main.py chi orchestrate.
# Cwd cua moi step luon = thu muc lab2_cnn/ (cha cua main.py),
# de paths "output/" va "../data" hoat dong dung nhu trong moi script.
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")

STEPS = [
    ("exp1_cgan_tinycnn",   "1. cGAN-MNIST + TinyCNN scratch"),
    ("gradcam_tinycnn",     "2. Grad-CAM TinyCNN (cGAN-MNIST)"),
    ("exp2_pgan_texcnn",    "3. PGAN-DTD + TexCNN scratch (baseline)"),
    ("exp3_pgan_resnet",    "4. PGAN-DTD + ResNet18 transfer"),
    ("exp4_biggan_resnet",  "5. BigGAN-128 + Imagenette + ResNet18 transfer"),
    ("gradcam_resnet",      "6. Grad-CAM ResNet18 (PGAN + BigGAN)"),
    ("cross_test",          "7. Cross-test BigGAN->PGAN"),
]


def run_step(mod_name, label, dry_run):
    bar = "=" * 70
    print(f"\n{bar}\n>> {label}\n{bar}", flush=True)
    if dry_run:
        print(f"  [dry-run] python src/{mod_name}.py")
        return 0
    t0 = time.time()
    rc = subprocess.call([sys.executable, os.path.join(SRC, f"{mod_name}.py")], cwd=ROOT)
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s  (exit code {rc})", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser(description="Run lab2_cnn pipeline")
    ap.add_argument("--only", nargs="+", help="chi chay nhung step nay (theo tu khoa, vi du: exp1 gradcam_resnet)")
    ap.add_argument("--dry-run", action="store_true", help="in lenh khong chay")
    ap.add_argument("--keep-going", action="store_true", help="khong stop khi step bi loi")
    args = ap.parse_args()

    selected = STEPS
    if args.only:
        selected = [(m, l) for m, l in STEPS if any(k in m for k in args.only)]
        if not selected:
            print(f"!!! Khong match step nao voi {args.only}", file=sys.stderr)
            sys.exit(2)

    failed = []
    for mod_name, label in selected:
        rc = run_step(mod_name, label, args.dry_run)
        if rc != 0:
            failed.append(mod_name)
            if not args.keep_going:
                print(f"\n!!! Step '{mod_name}' that bai (exit {rc}). Stop.", file=sys.stderr)
                sys.exit(rc)

    print("\n" + "=" * 70)
    if failed:
        print(f"Done with {len(failed)} failure(s): {failed}")
        sys.exit(1)
    print(f"Done. {len(selected)} steps OK.")


if __name__ == "__main__":
    main()
