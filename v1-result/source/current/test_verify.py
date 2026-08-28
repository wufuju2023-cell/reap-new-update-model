import argparse,ast,contextlib,hashlib,importlib.util,io,json,os,stat,sys,tempfile,unittest,zipfile
from pathlib import Path
from unittest import mock
s=importlib.util.spec_from_file_location('delivery_verify',Path(__file__).with_name('verify.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
spec=importlib.util.spec_from_file_location('delivery_reproduce',Path(__file__).with_name('reproduce.py'))
rep=importlib.util.module_from_spec(spec);spec.loader.exec_module(rep)

class VerifyTest(unittest.TestCase):
    def make(self,root,names=('a.py',)):
        with zipfile.ZipFile(root/'source.zip','w') as z:
            for name in names:z.writestr(name,b'pass\n')
        raw=(root/'source.zip').read_bytes()
        data={'archive':{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},
              'files':{name:{'bytes':5,'sha256':hashlib.sha256(b'pass\n').hexdigest()} for name in names}}
        (root/'manifest.json').write_text(json.dumps(data))
    def test_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);self.make(r)
            self.assertTrue(m.verify(r,r/'new')['verified'])
            self.assertEqual((r/'new/a.py').read_bytes(),b'pass\n')
            with self.assertRaises(FileExistsError):m.verify(r,r/'new')
    def test_archive_corruption(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);self.make(r);(r/'source.zip').write_bytes(b'bad')
            with self.assertRaises(ValueError):m.verify(r,r/'new')
            self.assertFalse((r/'new').exists())
    def test_unsafe_members(self):
        for names in [('a.py','A.py'),('../escape',),('/absolute',),('C:/x',),('a\\b',),('a./b',),
                      ('CON.txt',),('x/LPT1.py',),('x?.py',),('a','a/b'),('a','A/b')]:
            with self.subTest(names=names),tempfile.TemporaryDirectory() as t:
                r=Path(t);self.make(r,names)
                with self.assertRaises(ValueError):m.verify(r,r/'new')
                self.assertFalse((r/'new').exists())

    def test_hash_then_archive_replacement_uses_verified_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);self.make(r);original_hash=hashlib.sha256;calls=[]
            def digest(data):
                if not calls:(r/'source.zip').write_bytes(b'replaced after read')
                calls.append(True);return original_hash(data)
            with mock.patch.object(m.hashlib,'sha256',side_effect=digest):
                self.assertTrue(m.verify(r,r/'new')['verified'])
            self.assertEqual((r/'new/a.py').read_bytes(),b'pass\n')

    def test_nonregular_member_rejected(self):
        for mode in [stat.S_IFLNK,stat.S_IFIFO,stat.S_IFDIR]:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as t:
                r=Path(t);self.make(r)
                info=zipfile.ZipInfo('a.py');info.create_system=3;info.external_attr=(mode|0o600)<<16
                with zipfile.ZipFile(r/'source.zip','w') as z:z.writestr(info,b'pass\n')
                d=json.loads((r/'manifest.json').read_bytes());raw=(r/'source.zip').read_bytes()
                d['archive']={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
                (r/'manifest.json').write_text(json.dumps(d))
                with self.assertRaises(ValueError):m.verify(r,r/'new')
                self.assertFalse((r/'new').exists())

    def test_linked_parent_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);self.make(r);(r/'outside').mkdir()
            try:(r/'link').symlink_to(r/'outside',target_is_directory=True)
            except OSError:self.skipTest('directory symlink unavailable')
            with self.assertRaises(ValueError):m.verify(r,r/'link/new')
            self.assertFalse((r/'outside/new').exists())

    def test_reparse_attribute_rejected(self):
        with mock.patch.object(Path,'lstat',return_value=type('Stat',(),{'st_mode':stat.S_IFDIR,'st_file_attributes':0x400})()):
            with self.assertRaises(ValueError):m.no_links(Path.cwd()/'new')


class ReproduceTest(unittest.TestCase):
    def args(self,root,mode='mixed',extra=()):
        return ['reproduce.py',mode,'--source-root',str(root),'--data-root',str(root/'data'),
                '--model-path',str(root/'model'),'--output',str(root/'out'),*extra]

    def invoke(self,argv):
        with mock.patch.object(sys,'argv',argv),mock.patch.object(sys,'path',list(sys.path)),contextlib.redirect_stdout(io.StringIO()) as out:
            rc=rep.main()
        return rc,out.getvalue()

    def test_gpu_dryrun_command_matches_actual_parsers(self):
        source=Path(__file__).resolve().parents[3]
        for mode in ['mixed','continual']:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as t:
                r=Path(t);(r/'model').mkdir();(r/'release').mkdir()
                extra=[] if mode=='mixed' else ['--source-release-root',str(r/'release'),'--source-release-sha256','a'*64]
                with mock.patch.object(rep,'check',return_value={}),mock.patch.object(rep.subprocess,'call') as run:
                    rc,out=self.invoke(self.args(r,mode,extra))
                    self.assertEqual(rc,0);run.assert_not_called()
                command=json.loads(out)['command'];self.assertFalse((r/'out.launch-intent.json').exists())
                self.assertEqual(command.count('--replay-dataset-sha256'),4)
                self.assertEqual(command.count('--mathlib-dataset-sha256'),3)
                self.assertEqual('--max-distance' in command,mode=='mixed')
                file=source/'containers/gpu'/('smoke_mixed_learner.py' if mode=='mixed' else 'smoke_continual_mixed_learner.py')
                if file.exists():
                    # Execute only the stdlib parser definition, never GPU code.
                    node=next(n for n in ast.parse(file.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='build_parser')
                    namespace={'argparse':argparse,'Path':Path,'__doc__':'parser contract'}
                    exec(compile(ast.Module(body=[node],type_ignores=[]),str(file),'exec'),namespace)
                    namespace['build_parser']().parse_args(command[4:])

    def test_existing_output_or_launch_evidence_refuses_gpu(self):
        for name in ['out','out.launch-intent.json','out.exit.json']:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as t:
                r=Path(t);(r/'model').mkdir();(r/name).write_text('original')
                with mock.patch.object(rep,'check',return_value={}),mock.patch.object(rep.subprocess,'call') as run,contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):self.invoke(self.args(r,extra=['--run']))
                    run.assert_not_called()
                self.assertEqual((r/name).read_text(),'original')

    def test_unknown_launch_retains_intent_and_blocks_retry(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);(r/'model').mkdir()
            with mock.patch.object(rep,'check',return_value={}),mock.patch.object(rep.subprocess,'call',side_effect=OSError('launch outcome unknown')) as run:
                with self.assertRaises(OSError):self.invoke(self.args(r,extra=['--run']))
                intent=(r/'out.launch-intent.json').read_bytes()
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):self.invoke(self.args(r,extra=['--run']))
                self.assertEqual(run.call_count,1)
                self.assertEqual((r/'out.launch-intent.json').read_bytes(),intent)

    def test_failed_dataset_check_prevents_launch(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);(r/'model').mkdir()
            with mock.patch.object(rep,'check',side_effect=ValueError('corrupt pinned input')),mock.patch.object(rep.subprocess,'call') as run:
                with self.assertRaises(ValueError):self.invoke(self.args(r,extra=['--run']))
                run.assert_not_called()
            self.assertFalse((r/'out.launch-intent.json').exists())

    def test_prepare_no_overwrite_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);p=r/'new';name='frozen/datasets/'+'a'*64+'/dataset.json'
            with mock.patch.object(rep,'evidence',side_effect=lambda stage:{name:b'{}'} if stage=='verified-replay' else {}),mock.patch.object(rep,'check',side_effect=ValueError('incomplete dataset')):
                with self.assertRaises(ValueError):rep.prepare(p)
                original=(p/'replay'/('a'*64)/'dataset.json').read_bytes()
                with self.assertRaises(FileExistsError):rep.prepare(p)
                self.assertEqual((p/'replay'/('a'*64)/'dataset.json').read_bytes(),original)

    def test_prepare_linked_parent_refuses_before_read(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);(r/'outside').mkdir()
            try:(r/'link').symlink_to(r/'outside',target_is_directory=True)
            except OSError:self.skipTest('directory symlink unavailable')
            with mock.patch.object(rep,'evidence') as read:
                with self.assertRaises(ValueError):rep.prepare(r/'link/new')
                read.assert_not_called()

    def test_evidence_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);d=r/'evidence/current/test';d.mkdir(parents=True)
            (d/'raw.zip').write_bytes(b'bad')
            (d/'manifest.json').write_text(json.dumps({'archive':{'bytes':3,'sha256':'0'*64},'files':{}}))
            with mock.patch.object(rep,'PACKAGE',r):
                with self.assertRaises(ValueError):rep.evidence('test')

if __name__=='__main__':unittest.main()
