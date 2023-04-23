import sys
from lockexclusive import configure_lock

configure_lock(maxinstances=1, message="", file=sys.argv[0])
from shellextools import sleep
import os
from ordered_set import OrderedSet
import atexit
import shutil

from list_all_files_recursively import get_folder_file_complete_path
import pickle
from hackyargparser import add_sysargv
import base64

config = sys.modules[__name__]
config.foldername = None


def rm_dir(path, dryrun=True):
    if not os.path.exists(os.path.join(path)):
        return False
    allfolders = {}
    for file in get_folder_file_complete_path(path):
        allfolders[file.folder] = ""
        try:
            if not dryrun:
                os.remove(file.path)
        except Exception as fe:
            pass
    try:
        for newp in reversed(sorted(list(allfolders.keys()), key=lambda x: len(x))):
            try:
                if not dryrun:
                    shutil.rmtree(newp)
            except Exception as fe:
                continue

        if not dryrun:
            shutil.rmtree(path)
    except Exception as fa:
        pass

    return True


@atexit.register
def cleanup():
    try:
        if config.foldername:
            rm_dir(path=config.foldername, dryrun=False)
    except Exception:
        pass


@add_sysargv
def get_all_files(foldername: str = "", sleeptime: float | int = 4):
    tmpf = foldername
    config.foldername = tmpf
    fileset = OrderedSet()
    if not os.path.exists(tmpf):
        os.makedirs(tmpf)
    oldlen = -1
    newlen = 0
    while oldlen != newlen:
        oldlen = newlen

        try:
            folders = [tmpf]
            allfi = get_folder_file_complete_path(folders)
            for a in allfi:
                try:
                    if a.ext == ".shellexinfo":
                        if len(a.path.split(os.sep)[-1].split(".")[0]) == 64:
                            fileset.add(a.path)
                except Exception as fe:
                    continue
            newlen = len(fileset)
        except Exception:
            pass

        sleep(sleeptime)
    allvarsread = []
    for f in fileset:
        with open(f, mode="rb") as fi:
            allvarsread.append(pickle.loads(fi.read()))
    output_data = base64.standard_b64encode(pickle.dumps(allvarsread))
    sys.stdout.flush()
    sys.stdout.buffer.write(b"STARTSTARTSTARTSTART")
    sys.stdout.buffer.write(output_data)
    return 0


if __name__ == "__main__":
    get_all_files()
