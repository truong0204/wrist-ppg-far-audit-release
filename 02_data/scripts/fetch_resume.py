"""Resumable download for servers that ignore HTTP Range (UCI static, checked 2026-11-21).

Streams from byte 0; the first `have` bytes are compared with the partial file already on disk (abort
on any mismatch), then new bytes are appended. Retries on network errors.

    python -I fetch_resume.py <url> <dest.part> [max_retries]
"""
import os, sys, time
import requests

url, dest = sys.argv[1], sys.argv[2]
retries = int(sys.argv[3]) if len(sys.argv) > 3 else 20
CH = 1 << 20
for attempt in range(1, retries + 1):
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    try:
        with requests.get(url, headers={"User-Agent": "research-client/0.1"}, stream=True, timeout=60) as r, \
                open(dest, "r+b" if have else "wb") as f:
            r.raise_for_status()
            pos = 0
            for buf in r.iter_content(CH):
                if pos < have:                       # verify the prefix we already hold
                    n = min(len(buf), have - pos)
                    f.seek(pos)
                    if f.read(n) != buf[:n]:
                        sys.exit(f"prefix mismatch at byte {pos}: server content changed; delete {dest}")
                    pos += n; buf = buf[n:]
                    if not buf:
                        continue
                f.seek(pos); f.write(buf); pos += len(buf)
                if pos // (50 * CH) != (pos - len(buf)) // (50 * CH):
                    print(f"{time.strftime('%H:%M:%S')} {pos / 1e6:.0f} MB", flush=True)
        print(f"complete: {pos} bytes", flush=True)
        sys.exit(0)
    except Exception as e:  # network errors -> retry from 0 (prefix is verified, not rewritten)
        print(f"attempt {attempt} failed at {os.path.getsize(dest) / 1e6:.0f} MB: {e!r}", flush=True)
        time.sleep(10)
sys.exit("gave up")
