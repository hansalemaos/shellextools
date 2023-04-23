import base64
import random
import subprocess
import sys
import hashlib
import os
from pickle import dumps,loads
from shellextools import sleep


def to_pickle_dill(v, path):
    dumped = dumps(v)
    with open(path, mode='wb') as f:
        f.write(dumped)


def _fornu():
    from . import loggax3

def get_all_selected_files(sleeptime: float | int=4.0):
    fi = sys._getframe(1)
    dct = fi.f_globals
    f=dct.get("__file__", "")

    filepath = os.path.dirname(f)
    fpa=(os.path.join(filepath, 'loggax3.py'))
    if os.path.exists(fpa):
        filepath=fpa
    else:
        filepath = os.path.dirname(__file__)
        fpa = (os.path.join(filepath, 'loggax3.py'))
        if os.path.exists(fpa):
            filepath = fpa

        else:
            filepath = os.path.dirname(__file__)
            fpa = (os.path.join(filepath, 'loggax3.exe'))
            if os.path.exists(fpa):
                filepath = fpa
            else:
                filepath = os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1])
                fpa = (os.path.join(filepath, 'loggax3.exe'))
                filepath = fpa

    mainfile=os.path.normpath(filepath)
    hash = hashlib.sha256((mainfile + f"folder").encode("utf-8", "ignore"))
    foldername = hash.digest().hex()
    tmpf = os.path.join(os.environ.get("TMP"), foldername)
    try:
        if not os.path.exists(tmpf):
            os.makedirs(tmpf)
    except Exception:
        pass
    hash2 = hashlib.sha256((repr(sys.argv)).encode("utf-8", "ignore"))
    filename = hash2.digest().hex() + '.shellexinfo'
    tmpfile = os.path.join(tmpf,filename)
    to_pickle_dill(sys.argv[1:], tmpfile)

    startupinfo = subprocess.STARTUPINFO()
    creationflags = 0 | subprocess.CREATE_NO_WINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    sleep(random.uniform(sleeptime/16, sleeptime/4))
    if mainfile.endswith('.exe'):
        p = subprocess.run([mainfile,'--foldername', tmpf, '--sleeptime', f'{sleeptime}'], capture_output=True,
                           startupinfo=startupinfo, creationflags=creationflags
                           )

    else:
        p=subprocess.run([sys.executable, mainfile,'--foldername', tmpf,'--sleeptime', f'{sleeptime}'],capture_output=True,
                         startupinfo=startupinfo,creationflags=creationflags
                         )

    try:
        res2=base64.standard_b64decode(p.stdout.strip().split(b'STARTSTARTSTARTSTART')[-1])
        res = loads(res2)
        if not res:
            raise ValueError

    except Exception as ba:
        try:
            sys.exit(0)
        finally:
            os._exit(0)

    return res
