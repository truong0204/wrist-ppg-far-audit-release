"""List / extract members of a (possibly still-downloading) zip by walking local file headers.

PPG-DaLiA's UCI archive is `ppg+dalia.zip` -> stored `data.zip` -> members. Walking local headers
lets the pilot start on the first subjects before the 2.7 GB download finishes. The final
pipeline re-extracts from the complete, SHA256-checked archive with the standard zipfile module.

    python -I stream_zip.py list    <zip.part>
    python -I stream_zip.py extract <zip.part> <outdir> <member-prefix> [...]
"""
import os, struct, sys, zlib

def walk(path, base=0):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        off = base
        while off + 30 <= size:
            f.seek(off); h = f.read(30)
            sig, _, flag, meth, _, _, crc, cs, us, nl, el = struct.unpack("<IHHHHHIIIHH", h)
            if sig != 0x04034B50:
                return
            name = f.read(nl).decode("utf-8", "replace"); extra = f.read(el)
            if cs == 0xFFFFFFFF or us == 0xFFFFFFFF:      # zip64 sizes in extra field
                i = 0
                while i + 4 <= len(extra):
                    hid, hl = struct.unpack("<HH", extra[i:i + 4])
                    if hid == 1:
                        vals = struct.unpack("<" + "Q" * (hl // 8), extra[i + 4:i + 4 + hl])
                        it = iter(vals)
                        if us == 0xFFFFFFFF: us = next(it)
                        if cs == 0xFFFFFFFF: cs = next(it)
                    i += 4 + hl
            if flag & 0x08:
                raise RuntimeError(f"{name}: data descriptor (sizes unknown in local header) not supported")
            data_off = off + 30 + nl + el
            yield dict(name=name, method=meth, csize=cs, usize=us, crc=crc, data_off=data_off,
                       complete=data_off + cs <= size)
            off = data_off + cs

def inner_base(path):
    m = next(walk(path))
    assert m["name"] == "data.zip" and m["method"] == 0, m
    return m["data_off"]

def extract(path, outdir, prefixes):
    for m in walk(path, inner_base(path)):
        if not m["complete"] or not any(m["name"].startswith(p) for p in prefixes) or m["name"].endswith("/"):
            continue
        dest = os.path.normpath(os.path.join(outdir, m["name"]))
        assert dest.startswith(os.path.normpath(outdir) + os.sep), "path traversal"
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(path, "rb") as f:
            f.seek(m["data_off"]); raw = f.read(m["csize"])
        data = raw if m["method"] == 0 else zlib.decompress(raw, -15)
        assert zlib.crc32(data) & 0xFFFFFFFF == m["crc"], f"CRC mismatch {m['name']}"
        open(dest, "wb").write(data)
        print("extracted", m["name"], len(data))

if __name__ == "__main__":
    cmd, path = sys.argv[1], sys.argv[2]
    if cmd == "list":
        for m in walk(path, inner_base(path)):
            print(("OK  " if m["complete"] else "PART"), m["usize"], m["name"])
    else:
        extract(path, sys.argv[3], sys.argv[4:])
