import sys, os
import traceback
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def main():
    import run  # uses your configs, utils, process as-is

    t0 = time.time()
    print("== CPG generation start ==")
    print("Joern dir      :", getattr(run, "configs").Create().joern_cli_dir)
    print("RAW dir/file   :", run.PATHS.raw, run.FILES.raw)
    print("Output (CPG)   :", run.PATHS.cpg)

    try:
        run.CPG_generator()
    except Exception as e:
        print("\n[ERROR] CPG_generator failed:")
        traceback.print_exc()
        sys.exit(1)

    dt = time.time() - t0
    print(f"\n== Done in {dt:.1f}s ==")

    # --- quick report ---
    from pathlib import Path
    cpg_dir = Path(run.PATHS.cpg)
    pkl_files = sorted(cpg_dir.glob("*_{}.pkl".format(run.FILES.cpg)))
    json_files = sorted(cpg_dir.glob("*.json"))
    bin_files  = sorted(cpg_dir.glob("*.bin"))

    print(f"BIN files : {len(bin_files)}")
    print(f"JSON files: {len(json_files)}")
    print(f"PKL files : {len(pkl_files)}")

    if pkl_files:
        import pandas as pd
        df = pd.read_pickle(pkl_files[0])
        print("\nSample of first PKL:")
        print(df.head(3))

if __name__ == "__main__":
    main()