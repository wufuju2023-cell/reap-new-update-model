"""Offline delivery integrity checks. These do not repeat GPU/Lean experiments.

Default is read-only. --refresh replaces only package-manifest.json after all
payload checks pass. Use a trusted, exclusively owned package while checking.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from urllib.parse import unquote, urlsplit

sys.dont_write_bytecode = True  # A read-only checker must not create its own pycache.
SPEC = importlib.util.spec_from_file_location("_delivery_source_verify", Path(__file__).with_name("verify.py"))
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)
SCHEMA = "reap.local-delivery-manifest.v2"
LIMIT = 64 * 1024 * 1024
LEGACY = (
    ("source/source-manifest.json", "source/source-snapshot.tar.gz"),
    ("evidence/raw-evidence-manifest.json", "evidence/raw-evidence.tar.gz"),
    ("evidence/cpu-delivery/raw-evidence-manifest.json", "evidence/cpu-delivery/raw-evidence.tar.gz"),
    ("evidence/multiround/raw-evidence-manifest.json", "evidence/multiround/raw-evidence.tar.gz"),
    ("docs/history/archive-manifest.json", "docs/history/20260827-documents.tar.gz"),
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("overflowed JSON float: " + value)
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant, parse_float=finite_float)


def member_name(name):
    p = PurePosixPath(name)
    if (not name or not p.parts or p.is_absolute() or name != p.as_posix()
            or ".." in p.parts or any(ord(c) < 32 or c in '\\:<>"|?*' for c in name)
            or any(x.endswith((" ", ".")) or re.fullmatch(
                r"(?i)(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\..*)?", x) for x in p.parts)
            or "__pycache__" in p.parts or p.suffix.lower() in (".pyc", ".pyo")):
        raise ValueError("unsafe/compiled member: " + name)
    return name


def entries(value):
    """Normalize historical dict/list manifests without accepting duplicate paths."""
    result = {}
    rows = value.items() if isinstance(value, dict) else ((x["path"], x) for x in value)
    folded = set()
    for name, meta in rows:
        member_name(name)
        if name.casefold() in folded:
            raise ValueError("duplicate manifest path: " + name)
        folded.add(name.casefold())
        size = meta.get("bytes", meta.get("size"))
        sha = meta.get("sha256")
        if type(size) is not int or size < 0 or not isinstance(sha, str) or not re.fullmatch("[0-9a-f]{64}", sha):
            raise ValueError("invalid manifest entry: " + name)
        result[name] = {"bytes": size, "sha256": sha}
    for name in result:
        if any(p.as_posix().casefold() in folded for p in PurePosixPath(name).parents):
            raise ValueError("manifest file/directory collision: " + name)
    return result


def check_bytes(raw, meta, label):
    if len(raw) != meta.get("bytes", meta.get("size")) or digest(raw) != meta["sha256"]:
        raise ValueError("size/SHA mismatch: " + label)


def archive_values(manifest, raw, kind):
    archive = manifest.get("archive", {"bytes": manifest.get("archive_bytes"), "sha256": manifest.get("archive_sha256")})
    check_bytes(raw, archive, "archive")
    expected = entries(manifest["files"])
    if sum(x["bytes"] for x in expected.values()) > 1024 * 1024 * 1024:
        raise ValueError("archive exceeds 1GiB uncompressed safety limit")
    if kind == "zip":
        return VERIFY.checked_zip(raw, expected, LIMIT)
    values = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for item in tar:
            name = member_name(item.name)
            if not item.isfile() or item.size > LIMIT or name in values or name not in expected:
                raise ValueError("unsafe/unexpected tar member: " + name)
            stream = tar.extractfile(item)
            content = stream.read(LIMIT + 1)
            check_bytes(content, expected[name], name)
            values[name] = content
    if set(values) != set(expected):
        raise ValueError("tar member set mismatch")
    return values


def inventory(root):
    result = {}
    VERIFY.no_links(root)
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            p = Path(directory) / name
            st = p.lstat()
            if stat.S_ISLNK(st.st_mode) or getattr(st, "st_file_attributes", 0) & 0x400:
                raise ValueError("package link/reparse point: " + str(p))
            if not (stat.S_ISDIR(st.st_mode) or stat.S_ISREG(st.st_mode)):
                raise ValueError("nonregular package entry: " + str(p))
            member_name(p.relative_to(root).as_posix())
        for name in files:
            p = Path(directory) / name
            key = p.relative_to(root).as_posix()
            if key != "package-manifest.json":
                raw = p.read_bytes()
                result[key] = {"bytes": len(raw), "sha256": digest(raw)}
    entries(result)
    return result


def current_document(name):
    return (name in ("README.md", "docs/README.md") or name.startswith(
        ("docs/current/", "source/current/", "evidence/current/")))


def scan_markdown(root, files, bash=False):
    external, errors, bash_results = [], [], []
    checked = 0
    executable = shutil.which("bash") if bash else None
    for name in files:
        if not name.lower().endswith(".md"):
            continue
        text = (root / name).read_text(encoding="utf-8-sig")
        # Fenced examples are examples, not document links.
        fenced = re.compile(r"(?ms)^(`{3,}|~{3,})([^\n]*)\n(.*?)^\1\s*$")
        for match in fenced.finditer(text):
            if bash and match.group(2).strip().lower() == "bash":
                if executable:
                    proc = subprocess.run([executable, "-n"], input=match.group(3), text=True,
                                          capture_output=True, timeout=15)
                    bash_results.append({"document": name, "returncode": proc.returncode, "stderr": proc.stderr})
                    if proc.returncode:
                        errors.append("bash syntax: " + name)
        prose = fenced.sub("", text)
        targets = re.findall(r"!?\[[^\]\n]*\]\(\s*(<[^>]+>|[^)]+)\)", prose)
        targets += re.findall(r"(?m)^\s*\[[^\]]+\]:\s*(<[^>]+>|\S+)", prose)
        for target in targets:
            target = target.strip()
            if target.startswith("<"):
                target = target[1:target.find(">")]
            else:
                target = re.split(r'\s+[\"\']', target, maxsplit=1)[0]
            target = unquote(target)
            parsed = urlsplit(target)
            if parsed.scheme:
                if parsed.scheme.lower() not in ("http", "https", "mailto"):
                    errors.append("dangerous/absolute link: " + name + " -> " + target)
                continue
            if target.startswith("/") or "\\" in target or any(ord(c) < 32 for c in target):
                errors.append("unsafe link: " + name + " -> " + target)
                continue
            if not parsed.path:
                continue
            path = (root / name).parent / parsed.path
            resolved = path.resolve()
            checked += 1
            if not resolved.is_relative_to(root):
                external.append({"document": name, "target": target})
                if current_document(name):
                    errors.append("current document escapes package: " + name + " -> " + target)
            elif not path.exists():
                errors.append("missing relative link: " + name + " -> " + target)
    return {"relative_paths_checked": checked, "historical_project_links": external,
            "anchor_fragments_checked": False, "errors": errors,
            "bash": {"requested": bash, "executable": executable,
                     "status": "checked" if executable else "unavailable" if bash else "not_requested",
                     "blocks": bash_results}}


def check_package(root, refresh=False, bash=False):
    root = VERIFY.no_links(Path(root).absolute()).resolve(strict=True)
    initial = inventory(root)
    report = {"schema_version": "reap.package-integrity-check.v1", "ok": False,
              "scope": "File integrity and local syntax only; no new Lean/GPU/experiment validation",
              "errors": [], "archives": [], "archive_json_members_parsed": 0, "refreshed": False}
    errors = report["errors"]
    def run(label, operation):
        try:
            return operation()
        except Exception as exc:
            errors.append(label + ": " + str(exc))
    report["current_source"] = run("current source", lambda: VERIFY.verify(root / "source/current"))
    def check_archive(manifest_path, archive_path, kind, loose=False):
        manifest = strict_json((root / manifest_path).read_bytes())
        values = archive_values(manifest, (root / archive_path).read_bytes(), kind)
        for name, raw in values.items():
            member_name(name)
            if name.lower().endswith(".json"):
                strict_json(raw)
                report["archive_json_members_parsed"] += 1
        if loose:
            base = (root / manifest_path).parent / "files"
            expected = manifest["loose_files"]
            actual = {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()}
            if actual != set(expected):
                raise ValueError("loose file set mismatch")
            for name, sha in expected.items():
                member_name(name)
                raw = (base / name).read_bytes()
                if name not in values or raw != values[name] or digest(raw) != sha:
                    raise ValueError("loose copy differs from raw: " + name)
        report["archives"].append({"archive": archive_path, "members": len(values),
                                   "sha256": digest((root / archive_path).read_bytes()), "loose_checked": loose})
    run("current source members", lambda: check_archive("source/current/manifest.json", "source/current/source.zip", "zip"))
    for manifest, archive in LEGACY:
        run(archive, lambda m=manifest, a=archive: check_archive(m, a, "tar"))
    stages = root / "evidence/current"
    for stage in sorted(stages.iterdir()):
        if stage.is_dir():
            prefix = stage.relative_to(root).as_posix()
            run(prefix, lambda p=prefix: check_archive(p + "/manifest.json", p + "/raw.zip", "zip", True))
    json_count = 0
    for name in initial:
        if name.lower().endswith(".json"):
            run(name, lambda n=name: strict_json((root / n).read_bytes()))
            json_count += 1
    report["loose_json_files_parsed"] = json_count
    links = run("Markdown", lambda: scan_markdown(root, initial, bash))
    report["markdown"] = links
    if links:
        errors.extend(links["errors"])
    final = inventory(root)
    if final != initial:
        errors.append("package files changed during check")
    path = root / "package-manifest.json"
    if not refresh:
        def check_manifest():
            manifest = strict_json(path.read_bytes())
            if entries(manifest["files"]) != final:
                raise ValueError("package manifest differs; use --refresh only after reviewing intended changes")
            report["manifest_schema"] = manifest.get("schema", manifest.get("schema_version"))
        run("package manifest", check_manifest)
    elif not errors:
        value = {"schema_version": SCHEMA, "scope": report["scope"], "files": final}
        fd, temporary = tempfile.mkstemp(prefix=".package-manifest-", suffix=".tmp", dir=root)
        try:
            with os.fdopen(fd, "w", encoding="utf8", newline="\n") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        report["refreshed"] = True
        report["manifest_schema"] = SCHEMA
    report["file_count_excluding_package_manifest"] = len(final)
    report["ok"] = not errors
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--bash", action="store_true", help="Also run available local bash -n on fenced bash blocks")
    args = parser.parse_args(argv)
    try:
        result = check_package(args.package, args.refresh, args.bash)
    except Exception as exc:
        result = {"ok": False, "refreshed": False, "errors": [type(exc).__name__ + ": " + str(exc)]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
