"""T6 replay of the dev persons to the board over UART (PLAN_T6 §3-§4). Protocol: 06_firmware/h32_e2/main/main.cc.
    05_experiments/t6_mcu/.venv/bin/python 05_experiments/t6_mcu/replay.py --port /dev/cu.usbserial-XXXX --mode A
    05_experiments/t6_mcu/.venv/bin/python 05_experiments/t6_mcu/replay.py --port /dev/cu.usbserial-XXXX --mode B --runs 10
    05_experiments/t6_mcu/.venv/bin/python 05_experiments/t6_mcu/replay.py --port /dev/cu.usbserial-XXXX --mode ping
A: full replay of e61, u7y, ngh (+ e61 a second time, determinism) -> 02_data/derived/t6/board_<sid>.npz, board_e61_rep2.npz
B: n runs, each after a reset (DTR/RTS auto-reset; otherwise press EN when asked), first 1,800 windows of each person
   -> 05_experiments/t6_mcu/results/B_runs.npz + B_info.json
"""
import argparse, json, pathlib, struct, sys, time
import numpy as np, serial

ROOT = pathlib.Path(__file__).resolve().parents[2]; D = ROOT / "02_data/derived/t6"; OUT = ROOT / "05_experiments/t6_mcu/results"
IDS = ("e61", "u7y", "ngh"); RES = struct.Struct("<IffBBIIIIIB"); BLK = struct.Struct("<BxHfffBBBB"); EXT = struct.Struct("<II256b")
DEADLINE_S = 15.0; MAX_RESEND = 5
ap = argparse.ArgumentParser(); ap.add_argument("--port", required=True); ap.add_argument("--mode", choices=("A", "B", "ping"), required=True)
ap.add_argument("--runs", type=int, default=10); ap.add_argument("--prefix", default="board"); ap.add_argument("--no-rep2", action="store_true");
ap.add_argument("--a-windows", type=int, default=None); ap.add_argument("--a-ids", nargs="+", default=list(IDS));
ap.add_argument("--b-windows", type=int, default=1800); ap.add_argument("--b-ids", nargs="+", default=list(IDS)); ap.add_argument("--b-out", default="B"); ap.add_argument("--baud", type=int, default=921600); a = ap.parse_args()


def ck(b): return sum(b) & 0xFFFF


def open_port():
    s = serial.Serial(); s.port = a.port; s.baudrate = a.baud; s.timeout = 5; s.dtr = False; s.rts = False; s.open(); return s


def reset(s):
    s.dtr = False; s.rts = True; time.sleep(0.15); s.rts = False; time.sleep(1.5); s.reset_input_buffer()


def read_exact(s, n):
    b = s.read(n)
    if len(b) != n:
        raise TimeoutError(f"timeout: got {len(b)} of {n} bytes")
    return b


def read_tag(s):
    buf = b""; t_end = time.monotonic() + DEADLINE_S          # stall guard: hard deadline, even if bytes keep coming
    while True:
        c = s.read(1)
        if not c or time.monotonic() > t_end:
            raise TimeoutError("no tag before deadline")
        buf = (buf + c)[-4:]
        if buf in (b"H32I", b"H32X", b"H32A", b"H32R", b"H32N"):
            return buf


def read_text(s):
    n = struct.unpack("<H", read_exact(s, 2))[0]; t = read_exact(s, n); c = struct.unpack("<H", read_exact(s, 2))[0]
    assert c == ck(t), "text checksum"; return json.loads(t)


def start(s):
    for _ in range(20):
        s.write(b"H32S")
        try:
            if read_tag(s) == b"H32I":
                return read_text(s)
        except (TimeoutError, AssertionError, ValueError):
            time.sleep(0.5); s.reset_input_buffer()
    raise RuntimeError("board did not answer H32S (flash the firmware; check port and baud)")


def stats(s):
    s.write(b"H32E"); assert read_tag(s) == b"H32X"; return read_text(s)


def replay(s, sid, n_win=None):
    z = np.load(D / f"eval_wild_{sid}.npz"); ppg, acc = z["ppg64"].astype("<f4"), z["acc32"].astype("<f4")
    n = int(z["n"]) if n_win is None else n_win
    hr = np.empty(n, np.float32); sq = np.empty(n, np.float32); st = np.empty(n, np.int8); t = np.empty((n, 5), np.uint32)
    lg = np.empty((n, 256), np.int8); hs = np.empty(n, np.uint32)
    blk = []; t_uart = []; w = 0; info = start(s); resends = 0
    for k in range(n + 3):
        pay = struct.pack("<I", k) + ppg[128 * k:128 * k + 128].tobytes() + acc[64 * k:64 * k + 64].reshape(-1).tobytes()
        frame = b"H32W" + pay + struct.pack("<H", ck(pay)); t0 = time.perf_counter(); s.write(frame); got = None
        for attempt in range(MAX_RESEND + 1):
            try:
                while True:
                    tag = read_tag(s)
                    if tag == b"H32N":
                        read_exact(s, 4); s.write(frame); continue
                    if tag == b"H32A":
                        sq_ = struct.unpack("<I", read_exact(s, 4))[0]
                        if sq_ == k: got = "A"; break
                        continue                                  # stale ack of an earlier resend
                    if tag == b"H32R":
                        body = read_exact(s, RES.size + BLK.size + EXT.size + 2)
                        if struct.unpack("<H", body[-2:])[0] != ck(body[:-2]):
                            s.write(frame); continue
                        ext = EXT.unpack(body[RES.size + BLK.size:-2])
                        if ext[0] != k: continue                  # stale reply of an earlier resend
                        got = body; break
                break
            except TimeoutError:
                resends += 1; s.reset_input_buffer(); s.write(frame)
                print(f"  {sid}: resend chunk {k} (attempt {attempt + 1})", flush=True)
        if got is None:
            raise RuntimeError(f"{sid}: no reply for chunk {k} after {MAX_RESEND} resends")
        if got == "A":
            continue
        body = got; win, h, q, stt, mon, *times, nb = RES.unpack(body[:RES.size]); assert win == w, (win, w)
        hr[w], sq[w], st[w] = h, q, stt; t[w] = times
        ext = EXT.unpack(body[RES.size + BLK.size:-2]); hs[w] = ext[1]; lg[w] = ext[2:]
        if nb:
            blk.append(BLK.unpack(body[RES.size:RES.size + BLK.size]))
        t_uart.append(time.perf_counter() - t0); w += 1
        if w % 2000 == 0:
            print(f"  {sid}: {w}/{n} windows", flush=True)
    return dict(hr=hr, sqi=sq, st=st, t_us=t, blk=np.array(blk, float), roundtrip_s=np.array(t_uart), logits=lg, in_hash=hs,
                resends=np.array(resends)), info, stats(s)


s = open_port()
if a.mode == "ping":
    reset(s); print(json.dumps(start(s), indent=1)); print(json.dumps(stats(s))); sys.exit()
if a.mode == "A":
    reset(s); meta = {}
    for sid, tag in [(i, i) for i in a.a_ids] + ([] if a.no_rep2 else [("e61", "e61_rep2")]):
        r, info, st_ = replay(s, sid, a.a_windows); np.savez(D / f"{a.prefix}_{tag}.npz", **r); meta[tag] = dict(info=info, end=st_)
        print(tag, "done", st_, flush=True)
    json.dump(meta, open(OUT / f"A_{a.prefix}_info.json", "w"), indent=1)
if a.mode == "B":
    T = []; infos = []
    for run in range(a.runs):
        reset(s)
        try:
            start(s)
        except RuntimeError:
            if not sys.stdin.isatty():
                raise
            input(f"run {run}: press EN on the board, then Enter"); time.sleep(1.5); s.reset_input_buffer()
        run_t = []
        for sid in a.b_ids:
            r, info, st_ = replay(s, sid, a.b_windows); run_t.append(r["t_us"])
        infos.append(dict(run=run, info=info, end=st_)); T.append(np.concatenate(run_t)); print("run", run, st_, flush=True)
    np.savez(OUT / f"{a.b_out}_runs.npz", t_us=np.stack(T)); json.dump(infos, open(OUT / f"{a.b_out}_info.json", "w"), indent=1)
