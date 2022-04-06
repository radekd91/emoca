"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""



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
