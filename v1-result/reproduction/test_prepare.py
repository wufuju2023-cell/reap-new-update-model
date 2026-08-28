"""Local preparation checks; no Lean, model, network or container required."""
import contextlib
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import prepare


class PreparationTests(unittest.TestCase):
    def test_real_archives_extract_and_detect_modified_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / 'new-run'
            report = prepare.prepare(output)
            self.assertEqual((report['source_files'], report['input_files']), (188, 5))
            self.assertFalse(report['experiment_executed'])
            self.assertTrue(prepare.check(output)['ok'])
            path = output / 'inputs' / prepare.INPUT_NAMES[0]
            path.write_bytes(path.read_bytes() + b'\n-- changed\n')
            with self.assertRaisesRegex(ValueError, 'content or file set'):
                prepare.check(output)

    def test_existing_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(prepare, 'payload') as load:
                with self.assertRaises(FileExistsError):
                    prepare.prepare(Path(temporary).resolve())
                load.assert_not_called()

    def test_bad_payload_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / 'new-run'
            with patch.object(prepare, 'payload', side_effect=ValueError('bad archive')):
                with self.assertRaisesRegex(ValueError, 'bad archive'):
                    prepare.prepare(output)
            self.assertEqual(list(output.parent.iterdir()), [])

    def test_failed_write_does_not_commit_final_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / 'new-run'
            with patch.object(prepare, 'payload', return_value={'source/example.py': b'pass\n'}), \
                 patch.object(prepare.os, 'fsync', side_effect=OSError('disk failure')), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(OSError, 'disk failure'):
                    prepare.prepare(output)
            self.assertFalse(output.exists())
            self.assertEqual(len(list(output.parent.iterdir())), 1)

    def test_output_inside_report_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside the report'):
            prepare.prepare(prepare.HERE / 'new-run-that-must-not-exist')

    def test_code_copy_must_match_original_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            fake = Path(temporary)
            (fake / 'code').mkdir()
            (fake / 'code/manifest.json').write_bytes(b'{}')
            with patch.object(prepare, 'HERE', fake):
                with self.assertRaisesRegex(ValueError, 'differs from the frozen source'):
                    prepare.payload()

    def test_readable_code_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fake = Path(temporary)
            shutil.copytree(prepare.HERE / 'code', fake / 'code')
            (fake / 'code/src/gpu_runtime/server.py').write_bytes(b'changed\n')
            with patch.object(prepare, 'HERE', fake):
                with self.assertRaisesRegex(ValueError, 'readable code differs'):
                    prepare.payload()


if __name__ == '__main__':
    unittest.main()
