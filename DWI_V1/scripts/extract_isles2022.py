"""Extract verified official ZIP once; refuse overwrite and path traversal."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'datasets/ISLES2022/raw'
DOC = ROOT / 'datasets/ISLES2022/documentation'


def main():
    metadata = json.loads((DOC / 'zenodo-record.json').read_text())
    expected = next(f for f in metadata['files'] if f['key'] == 'ISLES-2022.zip')
    archive = RAW / expected['key']
    with archive.open('rb') as stream:
        if hashlib.file_digest(stream, 'md5').hexdigest() != expected['checksum'].split(':')[1]:
            raise RuntimeError('Archive MD5 does not match official record')
    manifest = []
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            target = (RAW / item.filename).resolve()
            if not target.is_relative_to(RAW.resolve()):
                raise RuntimeError(f'Unsafe archive path: {item.filename}')
            if target.exists() and not item.is_dir():
                raise FileExistsError(f'Refusing to overwrite {target}')
        for item in zipped.infolist():
            target = RAW / item.filename
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(item) as source, target.open('xb') as dest:
                shutil.copyfileobj(source, dest)
            with target.open('rb') as stream:
                checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
            manifest.append(dict(path=item.filename, size=item.file_size, sha256=checksum))
            if target.name.lower().startswith(('readme', 'license', 'dataset_description')):
                copied = DOC / 'archive_documents' / item.filename
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, copied)
    (DOC / 'raw-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Extracted {len(manifest)} files; all ZIP CRCs checked; raw SHA256 manifest saved.')


if __name__ == '__main__':
    main()
