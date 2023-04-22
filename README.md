# Adds Python functions/methods to the Windows context menu

## pip install shellextools 

### Here is an example:

https://github.com/hansalemaos/rc_pictools

[![](https://i.ytimg.com/vi/EsSrjG5vNpY/oar2.jpg?sqp=-oaymwEaCJUDENAFSFXyq4qpAwwIARUAAIhCcAHAAQY=&rs=AOn4CLDG3OahMcwdMtadJPwRe9lQvviQWA)](https://www.youtube.com/shorts/EsSrjG5vNpY)


## Create a pyw file 

```python
from PIL import Image
from hackyargparser import add_sysargv
from shellextools import (
    format_folder_drive_path_backslash,
    add_multicommands_files,
    change_file_extension,
)
import sys


@add_sysargv
def main(path: str = "", action: str = ""):
    path = format_folder_drive_path_backslash(path)
    img = Image.open(path)

    if action == "convert2ico":
        iconfi = change_file_extension(path=path, extension="ico")
        img.resize((512, 512)).save(iconfi)
    if action == "convert2gray":
        iconfi = change_file_extension(path=path, prefix="gray_", extension="png")
        img.convert("L").save(iconfi)
    if action == "convert2bw":
        iconfi = change_file_extension(path=path, prefix="bw_", extension="png")
        img.convert("1").save(iconfi)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        futurnameofcompiledexe = "pictools.exe"
        multicommands = [
            {
                "mainmenuitem": "PicTools",
                "submenu": "Convert to .ico",
                "folderinprogramdata": "RCTools",
                "filetypes": ["bmp", "png", "jpg", "jpeg"],
                "additional_arguments": "--action convert2ico",
            },
            {
                "mainmenuitem": "PicTools",
                "submenu": "Convert to grayscale",
                "folderinprogramdata": "RCTools",
                "filetypes": ["bmp", "png", "jpg", "jpeg"],
                "additional_arguments": "--action convert2gray",
            },
            {
                "mainmenuitem": "PicTools",
                "submenu": "Convert to bw",
                "folderinprogramdata": "RCTools",
                "filetypes": ["bmp", "png", "jpg", "jpeg"],
                "additional_arguments": "--action convert2bw",
            },
        ]
        add_multicommands_files(multicommands, futurnameofcompiledexe)
    else:
        main()



```



## Compile it with nutika 

```python
from nutikacompile import compile_with_nuitka

wholecommand = compile_with_nuitka(
    pyfile=r"C:\ProgramData\anaconda3\envs\nu\pictools.pyw",
    icon=r"C:\Users\hansc\Downloads\cova.jpg",
    disable_console=True,
    file_version="1.0.0.0",
    onefile=True,
    outputdir="c:\\nuitkapictoicon",
    addfiles=[
    ],
    delete_onefile_temp=False,  # creates a permanent cache folder
    needs_admin=True,
    arguments2add="--msvc=14.3 --noinclude-numba-mode=nofollow",
)


```

