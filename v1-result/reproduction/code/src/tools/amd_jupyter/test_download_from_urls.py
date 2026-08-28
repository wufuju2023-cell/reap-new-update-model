from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError, URLError

from tools.amd_jupyter import download_from_urls as download


class Response(io.BytesIO):
    def __init__(self, content, status=206, *, begin=0, total=None, declared_size=None):
        super().__init__(content)
        self.status = status
        size = len(content) if declared_size is None else declared_size
        total = begin + size if total is None else total
        self.headers = {"Content-Length": str(size)}
        if status == 206:
            self.headers["Content-Range"] = f"bytes {begin}-{begin + size - 1}/{total}"


class Opener:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class DownloadFromUrlsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "model"
        self.output.mkdir()
        self.name = "tokenizer.json"
        self.content = b"small-fixture-payload"
        self.entry = {"size": len(self.content), "lfs_sha256": hashlib.sha256(self.content).hexdigest()}
        self.url = "https://us.aws.cdn.hf.co/public/object?signature=PRIVATE_SENTINEL"
        self.progress = download.Progress(self.root / "progress.json", {self.name: self.entry})
        self.partial = self.output / (self.name + ".partial")
        self.final = self.output / self.name

    def run_transfer(self, opener):
        return download.transfer_one(self.name, self.entry, self.url, self.output, self.progress,
                                     opener=opener, sleep=lambda _: None)

    def assert_complete(self):
        self.assertEqual(self.final.read_bytes(), self.content)
        self.assertFalse(self.partial.exists())
        progress = json.loads(self.progress.path.read_text())
        self.assertEqual(progress["completed"], [self.name])
        self.assertTrue(progress["files"][self.name]["sha256_verified"])
        self.assertNotIn("PRIVATE_SENTINEL", self.progress.path.read_text())
        self.assertFalse((self.output / "reap-model-lock.json").exists())

    def test_complete_206_hashes_and_renames(self):
        opener = Opener(Response(self.content))
        self.assertTrue(self.run_transfer(opener))
        self.assert_complete()
        self.assertEqual(opener.requests[0][0].get_header("Range"), "bytes=0-")
        self.assertEqual(opener.requests[0][1], 30)

    def test_resume_checks_range_and_appends_only_remaining_bytes(self):
        self.partial.write_bytes(self.content[:5])
        opener = Opener(Response(self.content[5:], begin=5, total=len(self.content)))
        self.assertTrue(self.run_transfer(opener))
        self.assert_complete()
        self.assertEqual(opener.requests[0][0].get_header("Range"), "bytes=5-")

    def test_200_ignoring_resume_resets_partial_instead_of_appending(self):
        self.partial.write_bytes(self.content[:5])
        self.assertTrue(self.run_transfer(Opener(Response(self.content, status=200))))
        self.assert_complete()

    def test_wrong_content_range_preserves_partial_and_fails(self):
        self.partial.write_bytes(self.content[:5])
        opener = Opener(Response(self.content))
        self.assertFalse(self.run_transfer(opener))
        self.assertEqual(self.partial.read_bytes(), self.content[:5])
        self.assertFalse(self.final.exists())
        self.assertEqual(len(opener.requests), 1)
        self.assertEqual(self.progress.data["files"][self.name]["error"]["code"], "Content_Range_mismatch")

    def test_truncated_response_retries_with_exact_partial_offset(self):
        opener = Opener(Response(self.content[:5], total=len(self.content), declared_size=len(self.content)),
                        Response(self.content[5:], begin=5, total=len(self.content)))
        self.assertTrue(self.run_transfer(opener))
        self.assert_complete()
        self.assertEqual(opener.requests[1][0].get_header("Range"), "bytes=5-")

    def test_wrong_sha256_keeps_partial_without_success(self):
        self.assertFalse(self.run_transfer(Opener(Response(b"x" * len(self.content)))))
        self.assertTrue(self.partial.exists())
        self.assertFalse(self.final.exists())
        self.assertEqual(self.progress.data["completed"], [])
        self.assertEqual(self.progress.data["files"][self.name]["error"]["code"], "official_LFS_SHA256_mismatch")

    def test_expired_signed_url_error_never_logs_url_query(self):
        opener = Opener(HTTPError(self.url, 403, "denied " + self.url, {}, None))
        self.assertFalse(self.run_transfer(opener))
        self.assertEqual(len(opener.requests), 1)
        text = self.progress.path.read_text()
        self.assertNotIn("PRIVATE_SENTINEL", text)
        self.assertNotIn("https://", text)
        self.assertEqual(self.progress.data["files"][self.name]["error"]["status"], 403)

    def test_network_retry_sanitizes_underlying_exception(self):
        opener = Opener(URLError(self.url), Response(self.content))
        self.assertTrue(self.run_transfer(opener))
        self.assert_complete()
        self.assertEqual(len(opener.requests), 2)

    def test_existing_verified_final_requires_no_download(self):
        self.final.write_bytes(self.content)
        opener = Opener()
        self.assertTrue(self.run_transfer(opener))
        self.assert_complete()
        self.assertEqual(opener.requests, [])

    def test_only_approved_https_host_and_no_embedded_credentials(self):
        for url in ("http://us.aws.cdn.hf.co/object", "https://mirror.invalid/object",
                    "https://user:password@us.aws.cdn.hf.co/object", "https://us.aws.cdn.hf.co:8443/object"):
            with self.subTest(url=url), self.assertRaises(download.TransferError):
                download.validate_url(url)

    def test_url_preparation_uses_HEAD_without_reading_redirect_body(self):
        from tools.amd_jupyter import prepare_model_urls as prepare

        class NoReadBody(io.BytesIO):
            def read(self, *_):
                raise AssertionError("HEAD preparation must never read a response body")

        response = HTTPError("https://huggingface.co/fixture", 302, "Found", {"Location": self.url}, NoReadBody())
        opener = Opener(response)
        self.assertEqual(prepare.head_url(self.name, opener=opener), self.url)
        request, timeout = opener.requests[0]
        self.assertEqual(request.get_method(), "HEAD")
        self.assertEqual(timeout, 15)
        self.assertEqual(request.full_url, f"https://huggingface.co/{download.REPO}/resolve/{download.REVISION}/{self.name}")
        self.assertEqual(len(opener.requests), 1)

    def test_url_preparation_enforces_platform_privacy_and_never_overwrites(self):
        import os
        import stat
        from unittest.mock import Mock
        from tools.amd_jupyter import prepare_model_urls as prepare
        manifest = {"schema_version": "reap.official-model-manifest.v1", "repo": download.REPO, "revision": download.REVISION,
                    "source_api": f"https://huggingface.co/api/models/{download.REPO}/revision/{download.REVISION}?blobs=true",
                    "files": {name: self.entry for name in download.LFS_NAMES}}
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        checksum = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        output = self.root / "private-urls.json"
        head = Mock(return_value=self.url)
        if os.name != "posix":
            # A Windows chmod call cannot supply the POSIX 0600 guarantee. The
            # correct behavior is refusal before any HEAD or private write.
            with self.assertRaisesRegex(download.TransferError, "output_filesystem_cannot_enforce_0600"):
                prepare.prepare_urls(manifest_path, checksum, output, head=head)
            head.assert_not_called()
            self.assertFalse(output.exists())
            original = b"existing-output-must-be-preserved"
            output.write_bytes(original)
            with self.assertRaises(FileExistsError):
                prepare.prepare_urls(manifest_path, checksum, output, head=head)
            head.assert_not_called()
            self.assertEqual(output.read_bytes(), original)
            return
        report = prepare.prepare_urls(manifest_path, checksum, output, head=head)
        self.assertEqual(head.call_count, 5)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        private = json.loads(output.read_text())
        self.assertEqual(set(private["urls"]), download.LFS_NAMES)
        self.assertIn("fetched_at", private)
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(report))
        download.load_inputs(manifest_path, checksum, output)
        original = output.read_bytes()
        with self.assertRaises(FileExistsError):
            prepare.prepare_urls(manifest_path, checksum, output, head=head)
        self.assertEqual(head.call_count, 5)
        self.assertEqual(output.read_bytes(), original)

    def test_url_preparation_rejects_a_new_redirect_domain_without_following(self):
        from tools.amd_jupyter import prepare_model_urls as prepare
        opener = Opener(HTTPError("https://huggingface.co/fixture", 302, "Found",
                                  {"Location": "https://different.cdn.invalid/object?secret=PRIVATE_SENTINEL"}, None))
        with self.assertRaisesRegex(download.TransferError, "unapproved_download_url"):
            prepare.head_url(self.name, opener=opener)
        self.assertEqual(len(opener.requests), 1)


if __name__ == "__main__":
    unittest.main()
