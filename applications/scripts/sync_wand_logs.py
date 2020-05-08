
def sync_if_not_synced(path):
    path = os.path.abspath(path)
    sync_file = os.path.abspath(os.path.join(path, "..", "sync_status.txt"))

    if not os.path.exists(sync_file):
        print("Sync status file '%s' not found" % sync_file)
        return

    with open(sync_file, 'r') as f:
        status = f.read()

    if "not_synced" in status:
        print("This run was not synced. Syncing now")
        cwd = os.getcwd()
        os.chdir(path)
        os.system("wandb sync")
        with open(sync_file, 'w') as f:
            f.write("synced\n")
        os.chdir(cwd)
        print("Sync finished")
    else:
        print("Already synced")



if __name__ == "__main__":
    import os, sys, glob
    from multiprocessing import Pool

    if len(sys.argv) < 2:
        print("Pass a path with the experiments")
        exit(0)

    path = sys.argv[1]

    from pathlib import Path

    paths_to_sync = []
    for path in Path(path).rglob('dryrun*'):
    # for path in Path(path).rglob('run*'):
        paths_to_sync += [str(path)]

    multi = True
    if multi:
        with Pool(5) as p:
            print(p.map(sync_if_not_synced, paths_to_sync))
    else:
        for path in paths_to_sync:
            sync_if_not_synced(path)
    print("Syncing finished")
