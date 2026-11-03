import os
from pathlib import PurePosixPath
import re
import stat


def identifier(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',value):
        raise ValueError('Invalid identifier')
    return value


def relative_file(value):
    if not isinstance(value,str) or not value or len(value)>200 or '\\' in value:
        raise ValueError('Invalid artifact path')
    p=PurePosixPath(value)
    if not p.parts or p.as_posix()!=value or '\x00' in value or p.is_absolute() or '..' in p.parts or any(part.startswith('.') for part in p.parts):
        raise ValueError('Artifact path escapes output directory')
    return value


def safe_files(root,aliases=None):
    result=[];total=0
    for p in root.rglob('*'):
        try:
            info=p.lstat()
            if stat.S_ISLNK(info.st_mode) and aliases and aliases.get(p.relative_to(root).as_posix())==os.readlink(p):continue
        except FileNotFoundError:
            # A running test may remove a temporary fixture after directory enumeration.
            continue
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):raise ValueError('Sandbox produced a link or special file')
        if stat.S_ISREG(info.st_mode):
            total+=info.st_size;result.append(p)
    if total>1024**3 or len(result)>256:raise ValueError('Sandbox output quota exceeded')
    return result
