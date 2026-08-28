"""Verify and optionally extract the current source, using only Python stdlib."""
import argparse,hashlib,io,json,os,re,stat,zipfile
from pathlib import Path,PurePosixPath

def no_links(path):
    """Reject existing links/reparse points in operator-owned output paths.

    The caller owns the destination exclusively; this is not a sandbox against
    a concurrent privileged process replacing ancestors after this check.
    """
    path=Path(path)
    if '..' in path.parts:raise ValueError('parent traversal in output path')
    path=path.absolute()
    for item in [*reversed(path.parents),path]:
        try:info=item.lstat()
        except FileNotFoundError:continue
        if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:
            raise ValueError('linked/reparse output path: '+str(item))
    return path

def checked_zip(raw, expected, max_member_bytes):
    """Read the exact bytes already hashed, with portable regular-file names."""
    values={}; folded=set()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        if len(z.infolist())!=len(expected):raise ValueError('member count differs')
        for info in z.infolist():
            name=info.filename;p=PurePosixPath(name)
            mode=stat.S_IFMT(info.external_attr>>16)
            if (not name or not p.parts or p.is_absolute() or '..' in p.parts
                    or name!=p.as_posix() or name.casefold() in folded
                    or any(ord(c)<32 or c in '\\:<>"|?*' for c in name)
                    or info.is_dir() or mode not in (0,stat.S_IFREG)
                    or not 0<=info.file_size<=max_member_bytes
                    or any(part.endswith((' ','.')) or re.fullmatch(
                        r'(?i)(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\..*)?',part)
                        for part in p.parts)):
                raise ValueError('unsafe or duplicate member: '+name)
            folded.add(name.casefold())
            if name not in expected:raise ValueError('unexpected member')
            b=z.read(info);entry=expected[name]
            if len(b)!=entry['bytes'] or hashlib.sha256(b).hexdigest()!=entry['sha256']:
                raise ValueError('member integrity mismatch: '+name)
            values[name]=b
    if set(values)!=set(expected):raise ValueError('missing member')
    for name in values:
        if any(parent.as_posix().casefold() in folded for parent in PurePosixPath(name).parents):
            raise ValueError('file/directory member collision')
    return values

def verify(directory, destination=None):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_bytes())
    archive=directory/'source.zip'
    raw=archive.read_bytes()
    if len(raw)!=manifest['archive']['bytes'] or hashlib.sha256(raw).hexdigest()!=manifest['archive']['sha256']:
        raise ValueError('archive size/hash mismatch')
    values=checked_zip(raw,manifest['files'],4*1024*1024)
    if destination is not None:
        destination=no_links(destination)
        if destination.exists() or destination.is_symlink():raise FileExistsError(destination)
        if not destination.parent.is_dir() or destination.parent.resolve()!=destination.parent:
            raise ValueError('destination parent must exist without symbolic links')
        destination.mkdir()
        for name,b in values.items():
            p=destination/name;p.parent.mkdir(parents=True,exist_ok=True)
            if not p.parent.resolve().is_relative_to(destination):raise ValueError('escaped destination')
            with p.open('xb') as stream:stream.write(b)
    return {'verified':True,'files':len(values),'archive_sha256':manifest['archive']['sha256'],
            'extracted_to':str(destination) if destination is not None else None}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract',type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parent,args.extract)))
