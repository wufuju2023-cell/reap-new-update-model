#!/usr/bin/env python3
"""Verify a source snapshot completely before optional extraction; stdlib only."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile

MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_CONTENT_BYTES = 32 * 1024 * 1024
MAX_FILES = 10000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def no_links(path):
    for component in (path, *path.parents):
        require(not component.is_symlink() and not (
            hasattr(component, "is_junction") and component.is_junction()), "symlink/junction path rejected")


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def safe_name(name):
    require(isinstance(name, str) and bool(name), "empty/non-string path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and path.as_posix() == name and "\\" not in name
            and ":" not in name and all(ord(char) >= 32 and ord(char) != 127 for char in name)
            and all(part not in (".", "..") for part in path.parts), "unsafe archive path")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    require(all(part == part.rstrip(" .") and part.split('.')[0].upper() not in reserved
                for part in path.parts), "nonportable archive path")
    return path


def verify(archive, manifest):
    archive, manifest = Path(archive), Path(manifest)
    for path in (archive, manifest):
        no_links(path)
        require(path.is_file(), "archive/manifest must be regular files")
    require(manifest.stat().st_size <= 4 * 1024 * 1024, "manifest too large")
    data = json.loads(manifest.read_bytes(), object_pairs_hook=unique_keys)
    require(isinstance(data, dict), "manifest must be an object")
    require(data.get("schema") == "reap.delivery-source-snapshot.v1", "unknown manifest schema")
    expected = data.get("files")
    require(isinstance(expected, dict) and 0 < len(expected) <= MAX_FILES, "invalid file list")
    require(data.get("file_count") == len(expected), "manifest file count mismatch")
    total, portable_names = 0, set()
    for name, item in expected.items():
        safe_name(name)
        require(name.casefold() not in portable_names, "case-colliding archive names")
        portable_names.add(name.casefold())
        require(isinstance(item, dict) and type(item.get("size")) is int and item["size"] >= 0,
                "invalid member size")
        require(re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256"))) is not None, "invalid member digest")
        total += item["size"]
        require(total <= MAX_CONTENT_BYTES, "source payload too large")
    for name in expected:
        require(not any(parent.as_posix().casefold() in portable_names for parent in PurePosixPath(name).parents
                        if parent.as_posix() != "."), "file/directory path collision")
    require(data.get("uncompressed_bytes") == total, "manifest content size mismatch")
    header = data.get("archive", {})
    require(isinstance(header, dict), "archive metadata must be an object")
    require(type(header.get("size")) is int and 0 < header["size"] <= MAX_ARCHIVE_BYTES, "invalid archive size")
    require(archive.stat().st_size == header["size"], "archive size mismatch")
    compressed = archive.read_bytes()
    require(hashlib.sha256(compressed).hexdigest() == header.get("sha256"), "archive SHA256 mismatch")
    payload, modes = {}, {}
    with tarfile.open(fileobj=io.BytesIO(compressed), mode="r|gz") as stream:
        for member in stream:
            name = member.name
            safe_name(name)
            require(member.isfile() and member.sparse is None, "only ordinary nonsparse files are allowed")
            require(name in expected and name not in payload, "unexpected/duplicate archive member")
            item = expected[name]
            require(member.size == item["size"], "member size mismatch: " + name)
            content = stream.extractfile(member).read(member.size + 1)
            require(len(content) == item["size"] and hashlib.sha256(content).hexdigest() == item["sha256"],
                    "member SHA256 mismatch: " + name)
            payload[name] = content
            modes[name] = 0o755 if member.mode & 0o111 else 0o644
    require(set(payload) == set(expected), "missing archive members")
    return data, payload, modes


def extract_new(destination, payload, modes):
    target = Path(destination).absolute()
    no_links(target)
    require(not target.exists(), "extraction directory already exists; never overwrite")
    require(target.parent.is_dir(), "extraction parent must already exist")
    target.mkdir(mode=0o700)
    no_links(target)
    for name, content in sorted(payload.items()):
        current = target
        parts = safe_name(name).parts
        for component in parts[:-1]:
            current = current / component
            no_links(current)
            if not current.exists():
                current.mkdir(mode=0o700)
            require(current.is_dir(), "extraction parent is not a directory")
        path = current / parts[-1]
        no_links(path)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, modes[name])
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == content, "extracted file mismatch")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(__file__).with_name("source-snapshot.tar.gz"))
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("source-manifest.json"))
    parser.add_argument("--extract", type=Path, metavar="NEW_DIRECTORY")
    args = parser.parse_args()
    try:
        data, payload, modes = verify(args.archive, args.manifest)
        target = extract_new(args.extract, payload, modes) if args.extract is not None else None
        print(json.dumps({"verified": True, "file_count": len(payload), "archive_sha256": data["archive"]["sha256"],
                          "extracted_to": str(target) if target else None, "image_built": False}))
        return 0
    except (ValueError, OSError, tarfile.TarError, EOFError, TypeError, KeyError) as error:
        print(json.dumps({"verified": False, "error": str(error), "image_built": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
