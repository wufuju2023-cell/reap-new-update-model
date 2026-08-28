"""Install pinned successful-replay evidence into a local content registry.

No network, model, or learning operations. Directory traversal uses no-follow
handles; Windows directory handles also deny deletion/rename while in use.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import errno
import json
import os
from pathlib import Path
import re
import stat
from typing import Iterator
import uuid


BUNDLE_FILES = frozenset({
    "dataset.json", "session.json", "result.json", "raw_tree.json", "observer.jsonl",
    "source.lean", "accepted-proof.lean", "historical-proof-receipt.json", "VerifiedReplay.lean",
    "historical-proof.stdout", "historical-proof.stderr", "plan.json", "replay.lean",
    "replay-receipt.json", "trace.json", "replay.stdout", "replay.stderr",
})


class DatasetStoreError(ValueError):
    pass


def _name(name: str) -> str:
    if not name or name in (".", "..") or "/" in name or "\\" in name or ":" in name:
        raise DatasetStoreError("registry names must be single ordinary path components")
    return name


def _absolute(path: Path) -> Path:
    if ".." in Path(path).parts:
        raise DatasetStoreError("parent traversal is forbidden")
    # resolve() would follow the links we need to reject.
    return Path(os.path.abspath(path))


if os.name == "nt":
    from ctypes import wintypes
    import msvcrt

    class _FileInfo(ctypes.Structure):
        _fields_ = [("attributes", wintypes.DWORD), ("creation", wintypes.FILETIME),
                    ("access", wintypes.FILETIME), ("write", wintypes.FILETIME),
                    ("volume", wintypes.DWORD), ("size_high", wintypes.DWORD),
                    ("size_low", wintypes.DWORD), ("links", wintypes.DWORD),
                    ("index_high", wintypes.DWORD), ("index_low", wintypes.DWORD)]

    _kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                    ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    _kernel.CreateFileW.restype = wintypes.HANDLE
    _kernel.GetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.POINTER(_FileInfo)]
    _kernel.GetFileInformationByHandle.restype = wintypes.BOOL
    _kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel.CloseHandle.restype = wintypes.BOOL

    def _win_open(path: Path, *, directory: bool, create: bool = False) -> int:
        access = 0x80 if directory else (0x40000000 if create else 0x80000000)
        # Deny delete/rename; for files deny concurrent writers as well.
        share = 3 if directory else 1
        flags = 0x00200000 | (0x02000000 if directory else 0)  # OPEN_REPARSE_POINT | BACKUP_SEMANTICS
        handle = _kernel.CreateFileW(str(path), access, share, None, 1 if create else 3, flags, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _FileInfo()
        if not _kernel.GetFileInformationByHandle(handle, ctypes.byref(info)):
            error = ctypes.WinError(ctypes.get_last_error()); _kernel.CloseHandle(handle); raise error
        if info.attributes & 0x400 or bool(info.attributes & 0x10) != directory:
            _kernel.CloseHandle(handle)
            raise DatasetStoreError("symlink/junction/reparse point or wrong file type rejected")
        if directory:
            return handle
        try:
            return msvcrt.open_osfhandle(handle, os.O_BINARY | (os.O_WRONLY if create else os.O_RDONLY))
        except BaseException:
            _kernel.CloseHandle(handle)
            raise


class _Directory:
    def __init__(self, path: Path, handle: int):
        self.path, self.handle = path, handle

    def close(self) -> None:
        if os.name == "nt":
            _kernel.CloseHandle(self.handle)
        else:
            os.close(self.handle)

    @contextmanager
    def child(self, name: str, *, create: bool = False) -> Iterator[_Directory]:
        name = _name(name)
        path = self.path / name
        if create:
            if os.name == "nt":
                os.mkdir(path, mode=0o700)
            else:
                os.mkdir(name, mode=0o700, dir_fd=self.handle)
        if os.name == "nt":
            handle = _win_open(path, directory=True)
        else:
            handle = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.handle)
        child = _Directory(path, handle)
        try:
            yield child
        finally:
            child.close()

    def read(self, name: str) -> bytes:
        name = _name(name)
        if os.name == "nt":
            fd = _win_open(self.path / name, directory=False)
        else:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.handle)
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise DatasetStoreError("registry evidence must be a regular file")
            data = handle.read()
            after = os.fstat(handle.fileno())
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise DatasetStoreError("registry file changed while being read")
            return data

    def write_new(self, name: str, data: bytes) -> None:
        name = _name(name)
        if os.name == "nt":
            fd = _win_open(self.path / name, directory=False, create=True)
        else:
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=self.handle)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())

    def sync(self) -> None:
        if os.name != "nt":
            os.fsync(self.handle)


@contextmanager
def safe_directory(path: Path, *, create: bool = False) -> Iterator[_Directory]:
    """Open every component without following links; retain all ancestor handles."""
    path = _absolute(path)
    opened: list[_Directory] = []
    try:
        root = Path(path.anchor)
        handle = (_win_open(root, directory=True) if os.name == "nt" else
                  os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW))
        opened.append(_Directory(root, handle))
        for component in path.parts[1:]:
            parent = opened[-1]
            component = _name(component)
            child_path = parent.path / component
            if create:
                try:
                    if os.name == "nt":
                        os.mkdir(child_path, mode=0o700)
                    else:
                        os.mkdir(component, mode=0o700, dir_fd=parent.handle)
                except FileExistsError:
                    pass  # Opening below still rejects links and non-directories.
            handle = (_win_open(child_path, directory=True) if os.name == "nt" else
                      os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=parent.handle))
            opened.append(_Directory(child_path, handle))
        yield opened[-1]
    finally:
        for directory in reversed(opened):
            directory.close()


def read_bundle(directory: Path) -> dict[str, bytes]:
    with safe_directory(directory) as opened:
        return {name: opened.read(name) for name in sorted(BUNDLE_FILES)}


def _publish_noreplace(root: _Directory, staging: str, digest: str) -> None:
    """Atomic directory publication, including refusal to replace empty targets."""
    if os.name == "nt":
        # Windows rename rejects an existing target, even an empty directory.
        os.rename(root.path / staging, root.path / digest)
    elif os.name == "posix":
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise DatasetStoreError("atomic no-replace directory rename is unavailable")
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(root.handle, os.fsencode(staging), root.handle, os.fsencode(digest), 1) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), digest)
    else:
        raise DatasetStoreError("unsupported atomic registry platform")


def install_verified_dataset(source: Path, dataset_root: Path, *, expected_sha256: str) -> dict:
    """Validate, copy only the fixed bundle, revalidate, then publish once.

    A failure leaves an unadvertised .staging directory for inspection, never a
    partially populated digest directory. We do not recursively delete unknown
    files or retry ambiguous publication. Existing destinations are read-only.
    """
    from .verified_trajectory import load_verified_dataset, sha
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
        raise DatasetStoreError("a canonical SHA256 content pin is required")
    source = _absolute(source)
    # Validate before creating any destination directories.
    dataset = load_verified_dataset(source, expected_sha256=expected_sha256)
    with safe_directory(source) as original, safe_directory(dataset_root, create=True) as root:
        destination = root.path / expected_sha256
        try:
            with root.child(expected_sha256):
                installed = load_verified_dataset(destination, expected_sha256=expected_sha256)
                if installed != dataset:
                    raise DatasetStoreError("existing registry content differs")
                return {"status": "existing", "dataset_sha256": expected_sha256, "path": str(destination)}
        except FileNotFoundError:
            # Only a missing directory is admissible; missing evidence inside
            # an existing directory must not trigger replacement.
            try:
                with root.child(expected_sha256):
                    raise DatasetStoreError("existing registry directory is incomplete")
            except FileNotFoundError:
                pass
        staging_name = ".staging-" + uuid.uuid4().hex
        staging_path = root.path / staging_name
        with root.child(staging_name, create=True) as staging:
            for name in sorted(BUNDLE_FILES):
                staging.write_new(name, original.read(name))
            staging.sync()
        # This second loader uses the same no-follow discipline. Source changes
        # during copying cannot publish a mixed or mismatched evidence bundle.
        copied = load_verified_dataset(staging_path, expected_sha256=expected_sha256)
        if copied != dataset:
            raise DatasetStoreError("staged dataset differs from validated source")
        _publish_noreplace(root, staging_name, expected_sha256)
        root.sync()
        return {"status": "installed", "dataset_sha256": expected_sha256, "path": str(destination)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(install_verified_dataset(args.source, args.dataset_root, expected_sha256=args.sha256)))


if __name__ == "__main__":
    main()
