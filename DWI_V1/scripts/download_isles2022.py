"""Download the pinned official release with bounded parallel ranges and MD5 checks."""
import concurrent.futures
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'datasets/ISLES2022/raw'
PARTS = ROOT / 'datasets/ISLES2022/processed'
DOC = ROOT / 'datasets/ISLES2022/documentation'


def md5(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'md5').hexdigest()


def main():
    record = json.loads((DOC / 'zenodo-record.json').read_text())
    for file in record['files']:
        target = RAW / file['key']
        expected = file['checksum'].split(':')[1]
        if target.exists():
            if target.stat().st_size == file['size'] and md5(target) == expected:
                print(f'Already verified: {target.name}', flush=True)
                continue
            raise RuntimeError(f'Refusing to overwrite unverified existing file: {target}')
        prefix = PARTS / 'download-prefix.part'
        offset = prefix.stat().st_size if target.suffix == '.zip' and prefix.exists() else 0
        pieces = [prefix] if offset else []
        ranges = []
        chunk = 32 * 1024 * 1024
        for start in range(offset, file['size'], chunk):
            end = min(file['size'] - 1, start + chunk - 1)
            piece = PARTS / f'{target.name}.{start}-{end}.part'
            pieces.append(piece)
            ranges.append((start, end, piece))

        def fetch(item):
            start, end, piece = item
            expected_size = end - start + 1
            for attempt in range(6):
                current = piece.stat().st_size if piece.exists() else 0
                if current == expected_size:
                    return
                if current > expected_size:
                    raise RuntimeError(f'Oversized partial: {piece}')
                request = urllib.request.Request(file['links']['self'], headers={'Range': f'bytes={start + current}-{end}'})
                try:
                    with urllib.request.urlopen(request, timeout=90) as response:
                        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start + current}-{end}/{file["size"]}':
                            raise RuntimeError('Server did not honor exact byte range')
                        with piece.open('ab') as stream:
                            shutil.copyfileobj(response, stream, 1024 * 1024)
                    if piece.stat().st_size == expected_size:
                        print(f'Completed {target.name} bytes {start}-{end}', flush=True)
                        return
                except Exception as exc:
                    print(f'Retry {attempt + 1}: {exc}', flush=True)
                    time.sleep(min(2 ** attempt, 30))
            raise RuntimeError(f'Download incomplete: {piece}')

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(fetch, ranges))
        assembled = PARTS / (target.name + '.assembled')
        with assembled.open('wb') as output:
            for piece in pieces:
                with piece.open('rb') as source:
                    shutil.copyfileobj(source, output, 1024 * 1024)
        if assembled.stat().st_size != file['size'] or md5(assembled) != expected:
            raise RuntimeError(f'Official checksum FAILED for {target.name}')
        assembled.rename(target)
        print(f'VERIFIED {target.name} md5:{expected}', flush=True)
        for piece in pieces:
            if piece.resolve().parent != PARTS.resolve() or piece.suffix != '.part':
                raise RuntimeError(f'Unexpected staging cleanup path: {piece}')
            piece.unlink()
    from datetime import datetime
    (DOC / 'download-complete.json').write_text(json.dumps(dict(completed_at=datetime.now().astimezone().isoformat(), record=record['id'], files=record['files']), indent=2))


if __name__ == '__main__':
    main()
