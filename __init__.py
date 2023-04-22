import ctypes
import inspect
import pathlib
import shlex
import subprocess
import sys
import threading
from collections import defaultdict, namedtuple
from copy import deepcopy
from time import strftime, time
from escape_windows_filepath import escape_windows_path
from flatten_any_dict_iterable_or_whatsoever import (
    fla_tu,
)
from flatten_everything import flatten_everything
from functools import reduce, partial
from isiter import isiter
from list_files_with_timestats import (
    get_folder_file_complete_path,
)
from tolerant_isinstance import isinstance_tolerant
from touchtouch import touch
from typing import Union, List, Optional
import isiter
import os
import pickle
import re
import shutil
import stat
import tempfile
from subprocess_alive import is_process_alive
from subprocesskiller import get_pro_properties, subprocess_timeout, kill_process_children_parents, kill_subprocs, kill_pid
from amiadmin import am_i_admin
from downloadunzip import extract,copy_folder_to_another_folder,zip_folder,download_and_extract
from list_all_files_recursively import get_folder_file_complete_path
from reggisearch import search_values
from ctypestoast import show_notification,show_notification_threaded

nested_dict = lambda: defaultdict(nested_dict)
MB_OK = 0x0
MB_OKCXL = 0x01
MB_YESNOCXL = 0x03
MB_YESNO = 0x04
MB_HELP = 0x4000

ICON_EXCLAIM = 0x30
ICON_INFO = 0x40
ICON_STOP = 0x10
fields_cor = "folder file path ext"
classname_cor = "files"

FilePathInfos = namedtuple(classname_cor, fields_cor)

fields_cor_ts = "folder file path ext modified modified_ts created created_ts"
classname_cor_ts = "files"

FilePathInfosTime = namedtuple(classname_cor_ts, fields_cor_ts)

forbiddennames = r"""(?:CON|PRN|AUX|NUL|COM0|COM1|COM2|COM3|COM4|COM5|COM6|COM7|COM8|COM9|LPT0|LPT1|LPT2|LPT3|LPT4|LPT5|LPT6|LPT7|LPT8|LPT9)"""
compregex = re.compile(
    rf"(^.*?\\?)?\b{forbiddennames}\b(\.?[^\\]*$)?", flags=re.I
)
forbiddenchars = [
    "<",
    ">",
    ":",
    '"',
    "/",
    "\\",
    "|",
    "?",
    "*",
]
allcontrols_s = (
    "\x00",
    "\x01",
    "\x02",
    "\x03",
    "\x04",
    "\x05",
    "\x06",
    "\x07",
    "\x08",
    "\x09",
    "\x0a",
    "\x0b",
    "\x0c",
    "\x0d",
    "\x0e",
    "\x0f",
    "\x10",
    "\x11",
    "\x12",
    "\x13",
    "\x14",
    "\x15",
    "\x16",
    "\x17",
    "\x18",
    "\x19",
    "\x1a",
    "\x1b",
    "\x1c",
    "\x1d",
    "\x1e",
    "\x1f",
)


class FlexiblePartialOwnName:
    r"""
    FlexiblePartial(
            remove_file,
            "()",
            True,
            fullpath_on_device=x.aa_fullpath,
            adb_path=adb_path,
            serialnumber=device,
        )

    """

    def __init__(
        self, func, funcname: str, this_args_first: bool = True, *args, **kwargs
    ):
        self.this_args_first = this_args_first
        self.funcname = funcname
        try:
            self.f = copy_func(func)
        except Exception:
            self.f = func
        try:
            self.args = copy_func(list(args))
        except Exception:
            self.args = args

        try:
            self.kwargs = copy_func(kwargs)
        except Exception:
            try:
                self.kwargs = kwargs.copy()
            except Exception:
                self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        newdic = {}
        newdic.update(self.kwargs)
        newdic.update(kwargs)
        if self.this_args_first:
            return self.f(*self.args, *args, **newdic)

        else:
            return self.f(*args, *self.args, **newdic)

    def __str__(self):
        return self.funcname

    def __repr__(self):
        return self.funcname


class Popen(subprocess.Popen):
    def __init__(
        self,
        args,
        bufsize=-1,
        executable=None,
        stdin=None,
        stdout=None,
        stderr=None,
        preexec_fn=None,
        close_fds=True,
        shell=True,
        cwd=None,
        env=None,
        universal_newlines=None,
        startupinfo=None,
        creationflags=0,
        restore_signals=True,
        start_new_session=False,
        pass_fds=(),
        *,
        group=None,
        extra_groups=None,
        user=None,
        umask=-1,
        encoding=None,
        errors=None,
        text=None,
        pipesize=-1,
        process_group=None,
        print_output=True,
        **kwargs,
    ):
        stdin = subprocess.PIPE
        stdout = subprocess.PIPE
        universal_newlines = False
        stderr = subprocess.PIPE
        # shell = False
        startupinfo = subprocess.STARTUPINFO()
        creationflags = 0 | subprocess.CREATE_NO_WINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        hastimeout = "timeout" in kwargs
        timeout = 0
        if hastimeout:
            timeout = kwargs["timeout"]

            del kwargs["timeout"]

        super().__init__(
            args,
            bufsize=bufsize,
            executable=executable,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            preexec_fn=preexec_fn,
            close_fds=close_fds,
            shell=shell,
            cwd=cwd,
            env=env,
            universal_newlines=universal_newlines,
            startupinfo=startupinfo,
            creationflags=creationflags,
            restore_signals=restore_signals,
            start_new_session=start_new_session,
            pass_fds=pass_fds,
            group=group,
            extra_groups=extra_groups,
            user=user,
            umask=umask,
            encoding=encoding,
            errors=errors,
            text=text,
            **kwargs,
        )
        if hastimeout:
            timer = threading.Timer(timeout, partial(callback_func, self.pid))
            timer.start()
        self.stdout_lines = []
        self.stderr_lines = []
        self._stdout_reader = StreamReader(self.stdout, self.stdout_lines)
        self._stderr_reader = StreamReader(self.stderr, self.stderr_lines)
        stdo = self._stdout_reader.start()
        stdee = self._stderr_reader.start()
        for stdo_ in stdo:
            self.stdout_lines.append(stdo_)
            if print_output:
                print(stdo_)
        for stde_ in stdee:
            self.stderr_lines.append(stde_)
            if print_output:
                print(stde_)

        if hastimeout:
            timer.cancel()
        self.stdout = b"".join(self.stdout_lines)
        self.stderr = b"".join(self.stderr_lines)

    def __exit__(self, *args, **kwargs):
        try:
            self._stdout_reader.stop()
            self._stderr_reader.stop()
        except Exception as fe:
            pass

        super().__exit__(*args, **kwargs)

    def __del__(self, *args, **kwargs):
        try:
            self._stdout_reader.stop()
            self._stderr_reader.stop()
        except Exception as fe:
            pass
        super().__del__(*args, **kwargs)


class StreamReader:
    def __init__(self, stream, lines):
        self._stream = stream
        self._lines = lines
        self._stopped = False

    def start(self):
        while not self._stopped:
            line = self._stream.readline()
            if not line:
                break
            yield line

    def stop(self):
        self._stopped = True


def copy_func(f):
    if callable(f):
        if inspect.ismethod(f) or inspect.isfunction(f):
            g = lambda *args, **kwargs: f(*args, **kwargs)
            t = list(filter(lambda prop: not ("__" in prop), dir(f)))
            i = 0
            while i < len(t):
                setattr(g, t[i], getattr(f, t[i]))
                i += 1
            return g
    dcoi = deepcopy([f])
    return dcoi[0]

def exit_app(status):
    try:
        sys.exit(status)
    finally:
        os._exit(status)


def callback_func(pid):
    Popen(f"taskkill /F /PID {pid} /T", shell=False)


def timer_thread(timer, pid):
    timer.start()
    timer.join()
    callback_func(pid)


def allow_long_path_windows():
    winr = r"""Windows Registry Editor Version 5.00
    [HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem]
    "LongPathsEnabled"=dword:00000001
    """
    tem = get_tmpfile(suffix=".reg")
    with open(tem, mode="w", encoding="utf-8") as f:
        f.write(winr)
    os.startfile(tem)


def get_tmpfile(suffix=".bin"):
    tfp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    filename = tfp.name
    filename = os.path.normpath(filename)
    tfp.close()
    touch(filename)
    return filename


def groupby_files_folder_link(
    seq, continue_on_exceptions=True, withindex=False, withvalue=True
):
    return groupBy(
        key=lambda x: "folder"
        if os.path.isdir(x)
        else "file"
        if os.path.isfile(x)
        else "link"
        if os.path.islink(x)
        else "unknown",
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_file_extension(
    seq, continue_on_exceptions=True, withindex=False, withvalue=True
):
    return groupBy(
        key=lambda x: str(pathlib.Path(x).suffix).lower(),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def write_utf8(path, data, endofline="\n"):
    touch(path)
    with open(path, mode="w", encoding="utf-8") as f:
        for _ in fla_tu([data]):
            f.write(str(_[0]))
            f.write(endofline)


def write_bytes(path, data, endofline=None):
    touch(path)
    isit = isiter(data)
    with open(path, mode="wb") as f:
        if isit:
            for _ in fla_tu([data]):
                f.write(_[0])
                if endofline is not None:
                    f.write(endofline)
        else:
            f.write(data)


def read_bytes(path):
    with open(path, mode="rb") as f:
        data = f.read()
    return data


def copy_file(src, dst):
    with open(dst, "wb") as outfile:
        with open(src, "rb") as infile:
            for line in infile:
                outfile.write(line)


def make_filepath_windows_comp(
    filepath,
    fillvalue="_",
    reduce_fillvalue=True,
    remove_backslash_and_col=False,
    spaceforbidden=True,
    other_to_replace=(";", ",", "[", "]", "`", "="),
    slash_to_backslash=True,
):
    if slash_to_backslash:
        filepath = filepath.replace("/", "\\")
    filepath = filepath.strip()
    filepath = reduce(lambda a, b: a.replace(b, fillvalue), allcontrols_s, filepath)
    if other_to_replace:
        filepath = reduce(
            lambda a, b: a.replace(b, fillvalue), other_to_replace, filepath
        )

    filepath = filepath.strip()
    if spaceforbidden:
        filepath = re.sub(r"\s", fillvalue, filepath)
    for c in forbiddenchars:
        if not remove_backslash_and_col:
            if c == ":":
                filepath = filepath[:2] + filepath[2:].replace(c, fillvalue)
                continue
            if c == "\\":
                filepath = re.sub(r"\\+", "\\\\", filepath)
                continue
        filepath = filepath.replace(c, fillvalue)
    filepath2 = "".join(
        [
            x if x != "" else "_"
            for x in (flatten_everything(compregex.findall(filepath)))
        ]
    )
    if filepath2 != "":
        filepath = filepath2
        re.sub(r"\\+\.", r"\\_.", filepath)

    filepath = filepath.strip().strip("\\").strip().strip(".")
    filepath = re.sub(r"\.+", r".", filepath)
    if reduce_fillvalue:
        filepath = re.sub(rf"{fillvalue}+", rf"{fillvalue}", filepath)
        if len(filepath) > 1:
            filepath = filepath.strip(fillvalue)
    if len(filepath) == 0:
        filepath = fillvalue
    filepath = os.path.normpath(filepath)
    return filepath


def wrapre(
    text: Union[str, bytes],
    blocksize: int,
    regexsep: Union[str, bytes] = r"[\r\n]",
    raisewhenlonger: bool = True,
    removenewlines_from_result: bool = False,
    *args,
    **kwargs,
) -> List[Union[str, bytes]]:
    """
    Splits a given `text` into blocks of size `blocksize`, using the `regexsep` pattern as the separator.

    If `raisewhenlonger` is True (default), raises a ValueError if any block is larger than `blocksize`.

    If `removenewlines_from_result` is True, removes any newline characters from the resulting blocks.

    *args and **kwargs are additional arguments that can be passed to the `regex.compile` function.

    Args:
        text (str/bytes): The text to be split into blocks.
        blocksize (int): The maximum size of each block.
        regexsep (str/bytes): The regular expression pattern used to separate the blocks. Defaults to r"[\r\n]".
        raisewhenlonger (bool, optional): Whether to raise an error if any block is larger than `blocksize`. Defaults to True.
        removenewlines_from_result (bool, optional): Whether to remove any newline characters from the resulting blocks. Defaults to False.
        *args: Additional arguments to be passed to the `regex.compile` function.
        **kwargs: Additional keyword arguments to be passed to the `regex.compile` function.

    Returns:
        list: A list of strings (or bytes, if `text` was a bytes object), where each element is a block of text of maximum size `blocksize`.

    Raises:
        ValueError: If `raisewhenlonger` is True and any block is larger than `blocksize`.

    """
    spannow = -1
    limit = blocksize
    allspansdone = []
    allf = text
    isbytes = isinstance(text, bytes)
    regexsepcom = re.compile(regexsep, *args, **kwargs)

    while allf:
        oldlenallf = len(allf)
        newlenaffl = oldlenallf
        for ini, x in enumerate(regexsepcom.finditer(allf)):
            spannowtemp = x.end()
            if spannowtemp < limit:
                spannow = spannowtemp
            else:
                allspansdone.append(allf[:spannow])
                allf = allf[spannow:]
                spannow = -1
                newlenaffl = len(allf)
                break
        if oldlenallf == newlenaffl:
            allspansdone.append(allf)
            if not isbytes:
                allf = ""
            else:
                allf = b""
    if not allspansdone:
        allspansdone.append(allf)

    if raisewhenlonger:
        if len([True for x in allspansdone if len(x) > limit]) != 0:
            raise ValueError(
                "Some blocks are bigger than the limit! Try again with another separator or a bigger limit!"
            )
    if removenewlines_from_result:
        if isbytes:
            newlinesbtypes = re.compile(rb"[\r\n]+")
            allspansdone = [newlinesbtypes.sub(b" ", x) for x in allspansdone]
        else:
            newlinesbstr = re.compile(r"[\r\n]+")

            allspansdone = [newlinesbstr.sub(" ", x) for x in allspansdone]
    return allspansdone


def get_free_filename(folder: str, fileextension: str, leadingzeros: int = 9) -> str:
    if not os.path.exists(folder):
        os.makedirs(folder)
    savefolder_downloads = str(folder).rstrip(r"/\\")
    compiledreg = re.compile(rf"^0{{0,{leadingzeros-1}}}", re.IGNORECASE)
    compiledreg_checkfile = re.compile(
        rf"^\d{{{leadingzeros}}}{fileextension}", re.IGNORECASE
    )
    newfilenumber = 0
    try:
        picklefiles = os.listdir(f"{savefolder_downloads}{os.sep}")
        picklefiles = [x for x in picklefiles if str(x).endswith(fileextension)]
        picklefiles = [
            x for x in picklefiles if compiledreg_checkfile.search(x) is not None
        ]
        picklefiles = [int(compiledreg.sub("", _.split(".")[0])) for _ in picklefiles]
        newfilenumber = max(picklefiles) + 1
    except Exception as Fehler:
        pass
    finalfile = os.path.normpath(
        path=(
            os.path.join(folder, str(newfilenumber).zfill(leadingzeros) + fileextension)
        )
    )
    return finalfile


def rm_dir(path, dryrun=True):
    if not os.path.exists(os.path.join(path)):
        return False
    allfolders = {}
    for file in get_folder_file_complete_path(path):
        print(f"Deleting file: {file.path} ", end="\r")
        allfolders[file.folder] = ""
        try:
            if not dryrun:
                os.remove(file.path)
        except Exception as fe:
            print(fe)
            pass
    try:
        for newp in reversed(sorted(list(allfolders.keys()), key=lambda x: len(x))):
            try:
                print(f"Deleting folder: {newp}", end="\r")
                if not dryrun:
                    shutil.rmtree(newp)
            except Exception as fe:
                print(fe)
                continue

        if not dryrun:
            shutil.rmtree(path)
    except Exception as fa:
        print(fa)

    return True


def tempfolder():
    tempfolder = tempfile.TemporaryDirectory()
    tempfolder.cleanup()
    if not os.path.exists(tempfolder.name):
        os.makedirs(tempfolder.name)

    return tempfolder.name, _get_remove_folder(tempfolder.name)


def tempfolder_and_files(fileprefix="tmp_", numberoffiles=1, suffix=".bin", zfill=8):
    tempfolder = tempfile.TemporaryDirectory()
    tempfolder.cleanup()
    allfiles = []

    for fi in range(numberoffiles):
        tempfile____txtlist = os.path.join(
            tempfolder.name, f"{fileprefix}_{str(fi).zfill(zfill)}{suffix}"
        )
        allfiles.append(tempfile____txtlist)
        touch(tempfile____txtlist)

    return (
        [(k, _get_remove_file(k)) for k in allfiles],
        tempfolder.name.split(os.sep)[-1],
        tempfolder.name,
    )


def _get_remove_folder(folder):
    return FlexiblePartialOwnName(rm_dir, f"rm_dir({repr(folder)})", True, folder)


def _get_remove_file(file):
    return FlexiblePartialOwnName(os.remove, f"os.remove({repr(file)})", True, file)


def concat_files(filenames, output):
    with open(output, "wb") as wfd:
        for f in filenames:
            with open(f, "rb") as fd:
                shutil.copyfileobj(fd, wfd)


def enableLUA_disableLUA(enable=True):
    command = ""
    if enable is False:
        command = rf"""reg.exe ADD HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System /v EnableLUA /t REG_DWORD /d 0 /f"""
    if enable is True:
        command = rf"""reg.exe ADD HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System /v EnableLUA /t REG_DWORD /d 1 /f"""
    os.system(command)


def is_file_being_used(f):
    if os.path.exists(f):
        try:
            os.rename(f, f)
            return False
        except OSError as e:
            return True
    return None


def read_pkl(filename):
    with open(filename, "rb") as f:
        data_pickle = pickle.load(f)
    return data_pickle


def write_pkl(object_, savepath):
    touch(savepath)
    with open(savepath, "wb") as fa:
        pickle.dump(object_, fa)


def update_edit_time_of_file(path):
    os.utime(path, None)


def is_file_read_only(path):
    r"""True - read-only, False -> writable
    is_file_read_only(r"F:\bat.png")"""
    if os.path.exists(path):
        if os.access(path, os.W_OK):
            return False
        else:
            return True
    return None


def get_filesize(path):
    if os.path.exists(path):
        return os.path.getsize(path)
    return None


def set_file_writeable(path):
    return os.chmod(path, stat.S_IWRITE)


def set_file_read_only(path):
    r"""set_file_read_only(path=r"F:\bat.png")"""
    return os.chmod(path, stat.S_IREAD)


def encode_filepath(path):
    return os.fsencode(path)


def decode_filepath(path):
    return os.fsdecode(path)


def read_and_decode(path, decodeformat="utf-8", on_encoding_errors="replace"):
    asbytes = read_bytes(path)
    return asbytes.decode(decodeformat, errors=on_encoding_errors)


def iterread_bytes(path, chunksize=8192):
    with open(path, mode="rb") as f:
        while chunk := f.read(chunksize):
            yield chunk


def iterread_text(path, encoding="utf-8", strip_newline=True, ignore_empty_lines=True):
    with open(path, mode="r", encoding=encoding) as f:
        for line in f.readlines():
            if strip_newline:
                line = line.rstrip()
            if ignore_empty_lines:
                if line == "":
                    continue
            yield line


def convert_to_normal_dict(di):
    if isinstance_tolerant(di, defaultdict):
        di = {k: convert_to_normal_dict(v) for k, v in di.items()}
    return di


def groupBy(key, seq, continue_on_exceptions=True, withindex=True, withvalue=True):
    indexcounter = -1

    def execute_f(k, v):
        nonlocal indexcounter
        indexcounter += 1
        try:
            return k(v)
        except Exception as fa:
            if continue_on_exceptions:
                return "EXCEPTION: " + str(fa)
            else:
                raise fa

    # based on https://stackoverflow.com/a/60282640/15096247
    if withvalue:
        return convert_to_normal_dict(
            reduce(
                lambda grp, val: grp[execute_f(key, val)].append(
                    val if not withindex else (indexcounter, val)
                )
                or grp,
                seq,
                defaultdict(list),
            )
        )
    return convert_to_normal_dict(
        reduce(
            lambda grp, val: grp[execute_f(key, val)].append(indexcounter) or grp,
            seq,
            defaultdict(list),
        )
    )


def get_timestamp(sep="_"):
    return (
        strftime(f"%Y{sep}%m{sep}%d{sep}%H{sep}%M{sep}%S")
        + f"{sep}"
        + (str(time()).split(".")[-1] + "0" * 10)[:6]
    )


def create_file_with_timestamp(
    folder=None, extension=".tmp", prefix="", suffix="", sep="_", create_file=False
):
    tsfile = get_timestamp(sep=sep)
    if folder is not None:
        tsfile = os.path.normpath(
            os.path.join(folder, f"{prefix}{tsfile}{suffix}{extension}")
        )
    else:
        tsfile = os.path.normpath(f"{prefix}{tsfile}{suffix}{extension}")
    if create_file:
        touch(tsfile)
    return tsfile


def create_folder_with_timestamp(
    folder, prefix="", suffix="", sep="_", create_folder=False
):
    tsfile = get_timestamp(sep=sep)
    tsfile = os.path.normpath(os.path.join(folder, f"{prefix}{tsfile}{suffix}"))
    if create_folder:
        if not os.path.exists(tsfile):
            os.makedirs(tsfile)
    return tsfile


def execute_subprocess_multiple_commands_with_timeout_bin2(
    cmd: Union[list, str],
    subcommands: Union[list, tuple, None, str] = None,
    end_of_printline: str = "",
    print_output: bool = False,
    timeout: Optional[float] = None,
    cwd: str = os.getcwd(),
    decodestdout=None,
    decodestdouterrors: str = "ignore",
    stderrfile: Optional[str] = None,
    stdoutfile: Optional[str] = None,
    create_no_window: bool = True,
    use_shlex: bool = False,
    **kwargs,
) -> list:
    r"""
    Executes a subprocess and runs one or more commands in it, with the option to add a timeout and exit keys.
    :param cmd: The command to be executed in the subprocess.
    :type cmd: str
    :param subcommands: Additional commands to run in the subprocess, as a list or tuple of strings or a single string. Defaults to None.
    :type subcommands: Union[list, tuple, None, str]
    :param exit_keys: If set, the process can be terminated by pressing the specified key combination (e.g. "ctrl+alt+x"). Defaults to None.
    :type exit_keys: Union[str, None]
    :param end_of_printline: The string to be printed at the end of each line of output. Defaults to "".
    :type end_of_printline: str
    :param print_output: Whether to print the output of the subprocess to the console. Defaults to True.
    :type print_output: bool
    :param timeout: The maximum amount of time to allow the subprocess to run before terminating it. Defaults to None.
    :type timeout: Optional[float]
    :param cwd: The working directory for the subprocess. Defaults to the current working directory.
    :type cwd: str
    :param decodestdout: The character encoding to use for decoding the output of the subprocess. Defaults to None.
    :type decodestdout: Optional[str]
    :param decodestdouterrors: The error handling mode to use for decoding the output of the subprocess. Defaults to "ignore".
    :type decodestdouterrors: str
    :param stderrfile: The file path to write standard error output to. Defaults to None.
    :type stderrfile: Optional[str]
    :param stdoutfile: The file path to write standard output to. Defaults to None.
    :type stdoutfile: Optional[str]
    :param create_no_window: Whether to create a new console window for the subprocess. Defaults to True.
    :type create_no_window: bool
    :param use_shlex: Whether to use the shlex module to split the command string into arguments. Defaults to False.
    :type use_shlex: bool
    :param pyinstaller_module_name: The name of the PyInstaller module to run in the subprocess. Defaults to None.
    :type pyinstaller_module_name: Optional[str]
    :param pyinstaller_entry: The name of the PyInstaller entry point to run in the subprocess. Defaults to None.
    :type pyinstaller_entry: Optional[str]
    :param argsforpyinstaller: Additional arguments to pass to the PyInstaller subprocess. Defaults to ().
    :type argsforpyinstaller: Tuple
    :param kwargs: Additional keyword arguments to pass to the subprocess.Popen() constructor.
    :type kwargs: Any
    :return: A list of
    """

    _startupinfofun = subprocess.STARTUPINFO()
    creationflags = 0
    if create_no_window:
        creationflags = creationflags | subprocess.CREATE_NO_WINDOW
        _startupinfofun.wShowWindow = subprocess.SW_HIDE
    if isinstance_tolerant(subcommands, str):
        subcommands = [subcommands]
    elif isinstance_tolerant(subcommands, tuple):
        subcommands = list(subcommands)
    popen = None
    t = None
    stoutputfile = None

    def run_subprocess(cmd):
        nonlocal t
        nonlocal popen
        nonlocal stderrfile
        nonlocal stoutputfile

        if isinstance_tolerant(stderrfile, None):
            DEVNULL = open(os.devnull, "wb")
        else:
            stderrfile = os.path.normpath(stderrfile)
            if not os.path.exists(stderrfile):
                touch(stderrfile)
            DEVNULL = open(stderrfile, "ab")
        if not isinstance_tolerant(stdoutfile, None):
            if not os.path.exists(stdoutfile):
                touch(stdoutfile)
            stoutputfile = open(stdoutfile, "ab")
        try:
            if use_shlex:
                if isinstance_tolerant(cmd, str):
                    cmd = shlex.split(cmd)

            popen = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                universal_newlines=False,
                stderr=DEVNULL,
                shell=False,
                cwd=cwd,
                creationflags=creationflags,
                startupinfo=_startupinfofun,
                **kwargs,
            )

            if not isinstance_tolerant(subcommands, None):
                for subcommand in subcommands:
                    if isinstance(subcommand, str):
                        subcommand = subcommand.rstrip("\n") + "\n"

                        subcommand = subcommand.encode()
                    else:
                        subcommand = subcommand.rstrip(b"\n") + b"\n"

                    popen.stdin.write(subcommand)

            popen.stdin.close()

            woutput = not isinstance_tolerant(stoutputfile, None)
            dodecode = not isinstance_tolerant(decodestdout, None)
            for stdout_line in iter(popen.stdout.readline, b""):
                try:
                    if woutput:
                        try:
                            stoutputfile.write(stdout_line)
                        except Exception as fe:
                            print(fe)
                    if dodecode:
                        stdout_line = stdout_line.decode(
                            decodestdout, decodestdouterrors
                        )

                    yield stdout_line
                except Exception as Fehler:
                    continue
            popen.stdout.close()
            return_code = popen.wait()
        except Exception as Fehler:
            try:
                popen.stdout.close()
                return_code = popen.wait()
            except Exception as Fehler:
                yield ""
        finally:
            if not isinstance_tolerant(stderrfile, None):
                try:
                    DEVNULL.close()
                except Exception as fe:
                    pass
            if not isinstance_tolerant(stdoutfile, None):
                try:
                    stoutputfile.close()
                except Exception as fe:
                    pass

    proxyresults = []
    keyex = False
    try:
        for proxyresult in run_subprocess(cmd):
            proxyresults.append(proxyresult)
            if print_output:
                try:
                    print(f"{proxyresult!r}", end=end_of_printline)
                    print("")
                except Exception:
                    pass

    except KeyboardInterrupt:
        keyex = True
        _killpro(popen)

    if not keyex:
        _killpro(popen)
    return proxyresults


def set_read_write(path):
    os.chmod(path, stat.S_IWRITE)


def _killpro(popen):
    DEVNULL = open(os.devnull, "wb")
    _startupinfo = subprocess.STARTUPINFO()
    _startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    try:
        try:
            subprocess.Popen(
                f"taskkill /F /PID {popen.pid} /T",
                stdin=DEVNULL,
                stdout=DEVNULL,
                universal_newlines=False,
                stderr=DEVNULL,
                shell=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
                startupinfo=_startupinfo,
            )
        except Exception as fe:
            pass
    finally:
        try:
            DEVNULL.close()
        except Exception:
            pass


def set_read_only(path):
    os.chmod(path, stat.S_IREAD)


def normp(path):
    return os.path.normpath(path)


def get_new_file_vars(folderinprogramdata, futurnameofcompiledexe):
    dataf = os.environ.get("PROGRAMDATA")
    cmdtools = os.path.normpath(os.path.join(dataf, folderinprogramdata))
    myfile = os.path.normpath(os.path.join(os.getcwd(), futurnameofcompiledexe))
    newpath= (
        os.path.normpath(os.path.join(cmdtools, futurnameofcompiledexe))
    )
    newpathescaped = escape_windows_path(
        os.path.normpath(os.path.join(cmdtools, futurnameofcompiledexe))
    )
    newpathuninstall = os.path.normpath(
        os.path.join(cmdtools, futurnameofcompiledexe[:-4] + "_uninstall.cmd")
    )
    if not os.path.exists(cmdtools):
        os.makedirs(cmdtools)
    return dataf, cmdtools, myfile, newpath, newpathescaped, newpathuninstall


def show_after_install(silentinstall, newpath, newpathuninstall):
    if not silentinstall:
        ctypes.windll.user32.MessageBoxA(
            0,
            b"Files written:\n"
            + b"\n".join([x.encode() for x in [newpath, newpathuninstall]]),
            b"Successfully installed!",
            MB_OK | ICON_INFO,
        )

        ctypes.windll.user32.MessageBoxA(
            0,
            b"If you want to uninstall the app, execute: \n"
            + newpathuninstall.encode(),
            b"Important!",
            MB_OK | ICON_INFO,
        )


def change_file_extension(path,extension='',prefix='',suffix=''):
    path=os.path.normpath(path)
    iconfi = (".".join(path.split(".")[:-1]))+suffix +'.' + extension.strip(' .')
    iconfi=iconfi.split(os.sep)
    iconfi[-1]=prefix+iconfi[-1]
    return f'{os.sep}'.join(iconfi)

def execute_or_install_silent(
    sys_argv, getfoldercontent_no_subdirs=False, getfoldercontent_subdirs=False
):
    execute = False
    install = False
    silent = False
    isbiggerone = len(sys_argv) > 1
    silentinstall = False
    isfolder = False
    isdrive = False
    isfile = False
    fpathjoined = ""
    fpath = ""
    foldercontent_no_subdirs = []
    foldercontent_subdirs = []

    if isbiggerone:
        silentinstall = "--silentinstall" in sys.argv

    if isbiggerone and not silentinstall:
        execute = True
    elif isbiggerone and silentinstall:
        install = True
        silent = True
    elif not isbiggerone:
        install = True

    if not isbiggerone:
        fpathjoined = os.path.normpath("".join(sys_argv[1:]).strip('" '))
        fpath = os.path.normpath(sys_argv[1].strip('" '))
        if os.path.ismount(fpath) or os.path.ismount(fpathjoined):
            isdrive = True
        elif os.path.isdir(fpath) or os.path.isdir(fpathjoined):
            isfolder = True
        elif os.path.isfile(fpath) or os.path.isdir(fpathjoined):
            isfile = True

    returnpath = fpathjoined if os.path.exists(fpathjoined) else fpath
    returnpathfolder = os.path.normpath(os.path.dirname(returnpath))
    pathparts = pathlib.PureWindowsPath(returnpath).parts
    drive = pathlib.Path(returnpath).drive
    return (
        execute,
        install,
        silent,
        returnpath,
        returnpathfolder,
        pathparts,
        drive,
        isfolder,
        isdrive,
        isfile,
    )

def add_multi_commands_to_drive_and_folder(futurnameofcompiledexe,multicommands,):
    multicommandslen = len(multicommands)
    cou=0
    showdialogend = False
    newpathuninstall=''
    uninstalldata=''
    for ini, commandict in enumerate(multicommands):
        if multicommandslen - 1 == ini:
            showdialogend = True

        mainmenuitem = commandict["mainmenuitem"]
        submenu = commandict["submenu"]
        folderinprogramdata = commandict["folderinprogramdata"]
        putdrive = commandict["add2drive"]
        putfolder = commandict["add2folder"]
        additional_arguments = commandict["additional_arguments"]
        if putdrive:
            uninstalldata, newpathuninstall=(
                create_menu_with_submenu_for_drives(
                    showdialogend,
                    folderinprogramdata,
                    mainmenuitem,
                    submenu,
                    futurnameofcompiledexe,
                    additional_arguments=additional_arguments,
                    loopnumber=cou,
                )
            )
            cou+=1
        if putfolder:
            uninstalldata, newpathuninstall=(
                create_menu_with_submenu_for_folders(
                    showdialogend,
                    folderinprogramdata,
                    mainmenuitem,
                    submenu,
                    futurnameofcompiledexe,
                    additional_arguments=additional_arguments,
                    loopnumber=cou,
                )
            )
            cou += 1
        if showdialogend:
            with open(newpathuninstall, mode='a', encoding="utf-8") as f:
                f.write(uninstalldata)
                f.write('\n')



def get_one_folder_up(path):
    if len(path.strip('\\/:" '))>1:
        path = f"{os.sep}".join((path).split(os.sep)[:-1])
    return (path)

def format_path_for_windows(path):
    fp = make_filepath_windows_comp(
        filepath=path,
        fillvalue="_",  # replacement of any illegal char
        reduce_fillvalue=True,  # */<> (illegal chars) -> ____ (replacement) -> _ (reduced replacement)
        remove_backslash_and_col=True,  # important for multiple folders
        spaceforbidden=True,  # '\s' -> _
        other_to_replace=(
            ";",
            ",",
            "[",
            "]",
            "`",
            "=",
            ":",
        ),  # other chars you don't want in the file path
        slash_to_backslash=False,  # replaces / with \\ before doing all the other replacements
    )
    return fp

def create_menu_with_submenu_for_drives(
    silentinstall,
    folderinprogramdata,
    mainmenuitem,
    submenu,
    futurnameofcompiledexe,
    additional_arguments="",
    loopnumber=0,
):
    uninstalldata, newpathuninstall =create_menu_with_submenu_for_folders(
        silentinstall,
        folderinprogramdata,
        mainmenuitem,
        submenu,
        futurnameofcompiledexe,
        Drive_or_Directory="Drive",
        additional_arguments=additional_arguments,
        loopnumber=loopnumber,
    )
    return uninstalldata, newpathuninstall

def format_folder_drive_path_backslash(path):
    path = path.strip('"\\/: ')
    if len(path) == 1:
        path = f"{path}:"
    return path


def create_menu_with_submenu_for_folders(
    silentinstall,
    folderinprogramdata,
    mainmenuitem,
    submenu,
    futurnameofcompiledexe,
    Drive_or_Directory="Directory",
    additional_arguments="",
    loopnumber=0,
):
    if check_if_admin() == False:
        return 1
    dataf, cmdtools, myfile, newpath, newpathescaped, newpathuninstall = get_new_file_vars(
        folderinprogramdata, futurnameofcompiledexe
    )
    subprocess.run(
        rf"""%systemroot%\system32\Reg.exe delete "HKCR\{Drive_or_Directory}\shell\{mainmenuitem}\shell\{submenu}" /f""",
        shell=True,
    )
    subprocess.run(
        rf"""%systemroot%\system32\Reg.exe delete "HKCR\{Drive_or_Directory}\shell\{mainmenuitem}\shell\{submenu}\command" /f""",
        shell=True,
    )
    subprocess.run(
        rf"""%systemroot%\system32\Reg.exe delete "HKCR\{Drive_or_Directory}\shell\{mainmenuitem}\shell\{submenu}" /f""",
        shell=True,
    )
    subprocess.run(
        rf"""%systemroot%\system32\Reg.exe delete "HKCR\{Drive_or_Directory}\shell\{mainmenuitem}\shell\{submenu}\command" /f""",
        shell=True,
    )
    subprocess.run(
        rf"""%systemroot%\system32\Reg.exe add "HKCR\{Drive_or_Directory}\shell\{mainmenuitem}" /v "MUIVerb" /t REG_SZ /d "{mainmenuitem}" /f""",
        shell=True,
    )
    subprocess.run(
        rf"""%systemroot%\system32\Reg.exe add "HKCR\{Drive_or_Directory}\shell\{mainmenuitem}" /v "SubCommands" /t REG_SZ /d "" /f""",
        shell=True,
    )
    subprocess.run(
        rf"""%systemroot%\system32\Reg.exe add "HKCR\{Drive_or_Directory}\shell\{mainmenuitem}\shell\{submenu}" /v "MUIVerb" /t REG_SZ /d "{submenu}" /f""",
        shell=True,
    )
    positionofitem = 1
    try:
        positionofitem = (
            len(
                subprocess.run(
                    rf"""%systemroot%\system32\Reg.exe query "HKCR\{Drive_or_Directory}\shell\{mainmenuitem}\shell""",
                    shell=True,
                    capture_output=True,
                )
                .stdout.decode("utf-8", "ignore")
                .strip()
                .splitlines()
            )
            + 1
        )
    except Exception as fe:
        pass
    subprocess.run(
        rf"""%systemroot%\system32\Reg.exe add "HKCR\{Drive_or_Directory}\shell\{mainmenuitem}\shell\{submenu}" /v "Position" /t REG_SZ /d "{positionofitem}" /f""",
        shell=True,
    )

    additional_arguments = additional_arguments.strip()
    if additional_arguments:
        additional_arguments = f" {additional_arguments}"
    # subprocess.run(
    #     rf"""%systemroot%\system32\Reg.exe add "HKCR\{Drive_or_Directory}\shell\{mainmenuitem}\shell\{submenu}\command" /ve /t REG_SZ /d "{newpathescaped} \"%1\"{additional_arguments}" /f""",
    #     shell=True,
    # )
    addtocommand1=r'\"'
    addtocommand0 = r'\"'
    if Drive_or_Directory == 'Drive':
        addtocommand0=''
        addtocommand1=''
    cm=rf"""%systemroot%\system32\reg.exe add "HKCR\{Drive_or_Directory}\shell\{mainmenuitem}\shell\{submenu}\command" /ve /t REG_SZ /d "\"{newpath}\" --path {addtocommand0}%1{addtocommand1}{additional_arguments}" /f"""
    #print(cm)
    subprocess.run(
        cm,
        shell=True,
    )
    if loopnumber == 0:
        shutil.copy(myfile, newpath)
    uninstalldata = f"\ndel {newpathescaped}\ndel {newpathuninstall}\n"
    if loopnumber == 0:
        writemode = "w"
    else:
        writemode = "a"
    with open(newpathuninstall, mode=writemode, encoding="utf-8") as f:
        f.write('\n')
        f.write(
            rf"""%systemroot%\system32\Reg.exe delete "HKCR\{Drive_or_Directory}\shell\{mainmenuitem}\shell" /f"""
        )
        f.write('\n')
        f.write(
            rf"""%systemroot%\system32\Reg.exe delete "HKCR\{Drive_or_Directory}\shell\{mainmenuitem}" /f""",
        )
        f.write('\n')
        f.write(
            rf"""%systemroot%\system32\Reg.exe delete "HKCR\{Drive_or_Directory}\shell\{mainmenuitem}\shell\{submenu}" /f"""
        )
        f.write('\n')
        f.write(
            rf"""%systemroot%\system32\Reg.exe delete "HKCR\{Drive_or_Directory}\shell\{mainmenuitem}\shell\{submenu}\command" /f""",
        )
        f.write('\n')
        f.write(
            rf"""%systemroot%\system32\Reg.exe delete "HKCR\{Drive_or_Directory}\shell\{mainmenuitem}\shell\{submenu}" /f"""
        )
        f.write('\n')
        f.write(
            rf"""%systemroot%\system32\Reg.exe delete "HKCR\{Drive_or_Directory}\shell\{mainmenuitem}\shell\{submenu}\command" /f"""
        )
        f.write('\n')
        f.write(
            fr'%systemroot%\system32\Reg.exe delete "HKCR\{Drive_or_Directory}\shell\{mainmenuitem}\shell\{submenu}\command"  /f'
        )
        #f.write(uninstalldata)
        f.write('\n')
    # if silentinstall:
    #     show_after_install(silentinstall, newpath, newpathuninstall)
    return uninstalldata,newpathuninstall


def check_if_admin():
    p = subprocess.run(
        r'''%systemroot%\system32\Reg.exe query "HKU\S-1-5-19\Environment"''',
        shell=True,
    )
    if p.returncode == 0:
        return True
    ctypes.windll.user32.MessageBoxA(
        0,
        b"Execute the file with admin rights to install the software",
        b"Admin rights",
        MB_OK | ICON_STOP,
    )
    return False

def add_multicommands_files(multicommands,futurnameofcompiledexe):
    silentinstall=True
    for ini, m in enumerate(multicommands):

        newpathuninstall, finaldelete = create_menu_with_submenu_for_specific_files(
            silentinstall,
            m['folderinprogramdata'],
            m['mainmenuitem'],
            m['submenu'],
            futurnameofcompiledexe,
            m['filetypes'],
            additional_arguments=m['additional_arguments'],
            loopnumber=ini,
        )
        if ini  == len(multicommands) - 1:
            with open(newpathuninstall, mode='a', encoding="utf-8") as f:
                f.write(finaldelete)

def create_menu_with_submenu_for_specific_files(
    silentinstall,
    folderinprogramdata,
    mainmenuitem,
    submenu,
    futurnameofcompiledexe,
    filetypes,
    additional_arguments="",
    loopnumber=0,
):
    fileswritten = []
    if check_if_admin() == False:
        return 1

    dataf, cmdtools, myfile, newpath, newpathescaped, newpathuninstall = get_new_file_vars(
        folderinprogramdata, futurnameofcompiledexe
    )
    addtocommand0=r'\"'
    addtocommand1=r'\"'
    additional_arguments = additional_arguments.strip()
    if additional_arguments:
        additional_arguments = f" {additional_arguments}"
    commandsraw = [
        rf"""%systemroot%\system32\Reg.exe add "HKCR\.FTYPE\shell\{mainmenuitem}" /v "MUIVerb" /t REG_SZ /d "{mainmenuitem}" /f""",
        rf"""%systemroot%\system32\Reg.exe add "HKCR\.FTYPE\shell\{mainmenuitem}" /v "SubCommands" /t REG_SZ /d "" /f""",
        rf"""%systemroot%\system32\Reg.exe add "HKCR\.FTYPE\shell\{mainmenuitem}\shell" /v "MUIVerb" /t REG_SZ /d "{submenu}" /f""",
        rf"""%systemroot%\system32\Reg.exe add "HKCR\.FTYPE\shell\{mainmenuitem}\shell\{submenu}\command" /ve /t REG_SZ /d "{newpathescaped} \"%1\"{additional_arguments}" /f""",
        rf"""%systemroot%\system32\Reg.exe add "HKCR\SystemFileAssociations\.FTYPE\shell\{mainmenuitem}" /v "MUIVerb" /t REG_SZ /d "{mainmenuitem}" /f""",
        rf"""%systemroot%\system32\Reg.exe add "HKCR\SystemFileAssociations\.FTYPE\shell\{mainmenuitem}" /v "SubCommands" /t REG_SZ /d "" /f""",
        rf"""%systemroot%\system32\Reg.exe add "HKCR\SystemFileAssociations\.FTYPE\shell\{mainmenuitem}\shell" /v "MUIVerb" /t REG_SZ /d "{submenu}" /f""",
        rf"""%systemroot%\system32\Reg.exe add "HKCR\SystemFileAssociations\.FTYPE\shell\{mainmenuitem}\shell\{submenu}\command" /ve /t REG_SZ /d "\"{newpath}\" --path {addtocommand0}%1{addtocommand1}{additional_arguments}" /f""",
    ]
    commandsrawdelete = [
        rf"""%systemroot%\system32\Reg.exe delete "HKCR\.FTYPE\shell\{mainmenuitem}" /f""",

        rf"""%systemroot%\system32\Reg.exe delete "HKCR\.FTYPE\shell\{mainmenuitem}\shell" /f""",
        rf"""%systemroot%\system32\Reg.exe delete "HKCR\.FTYPE\shell\{mainmenuitem}\shell\{submenu}" /f""",
        rf"""%systemroot%\system32\Reg.exe delete "HKCR\SystemFileAssociations\.FTYPE\shell\{mainmenuitem}\shell\{submenu}" /f""",
        rf"""%systemroot%\system32\Reg.exe delete "HKCR\SystemFileAssociations\.FTYPE\shell\{mainmenuitem}" /f""",


    ]
    uninstalldatalist = []
    for fi in filetypes:
        for c in commandsrawdelete:
            command2add = c.replace(rf"\.FTYPE{os.sep}", rf"\.{fi}{os.sep}")
            uninstalldatalist.append(command2add)

    commands = []
    for fi in filetypes:
        for c in commandsraw:
            command2add = c.replace(rf"\.FTYPE{os.sep}", rf"\.{fi}{os.sep}")
            commands.append(command2add)

    for c in commands:
        subprocess.run(c, shell=True)

    shutil.copy(myfile, newpath)
    fileswritten.append(newpath)
    uninstalldata = "\n".join(uninstalldatalist)
    finaldelete = f"\ndel {newpathescaped}\ndel {newpathuninstall}\n"
    if loopnumber == 0:
        writemode = "w"
    else:
        writemode = "a"
    with open(newpathuninstall, mode=writemode, encoding="utf-8") as f:
        f.write(uninstalldata)
    fileswritten.append(newpathuninstall)
    #show_after_install(silentinstall, newpath, newpathuninstall)

    return newpathuninstall,finaldelete
