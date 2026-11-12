"""Meeting recorder window: records the default mic + what you hear, then runs ./process.
Opened by ~/.local/bin/meeting-rec (Super+Shift+M). Stop = Enter, the hotkey again, or closing
the window; all three save the recording. Saved as stereo WAV: left = mic, right = laptop audio."""
import audioop, collections, json, math, os, select, signal, subprocess, sys, termios, threading, time, tty, wave
import urllib.parse
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PIDFILE = os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "meeting-rec.pid")
RATE, BLOCK = 16000, 3200  # bytes per read = 100 ms of 16-bit mono
FG, BG = "\x1b[38;2;177;185;249m", "\x1b[48;2;0;0;0m"  # #b1b9f9 on black
W = 52
stop = hup = False


def on_signal(sig, _):
    global stop, hup
    stop, hup = True, hup or sig == signal.SIGHUP


def parec(device):
    return subprocess.Popen(["parec", f"--device={device}", f"--rate={RATE}", "--channels=1",
                             "--format=s16le", "--raw", "--latency-msec=100"], stdout=subprocess.PIPE)


def devices():
    """[(source to record, short name)] for the current default mic and output.
    The output is recorded via its '.monitor': @DEFAULT_MONITOR@ falls back to the mic on PipeWire 0.3.48."""
    def short(d):
        # built-in card: "sof-hda-dsp Speaker + Headphones" -> active port, e.g. "Speaker" or "Headphones"
        if d["description"].startswith(d["properties"].get("alsa.card_name", "\0")):
            return next((p["description"] for p in d.get("ports", []) if p["name"] == d.get("active_port")),
                        d["description"])
        return d["description"]  # USB / Bluetooth: product name, e.g. "Beoplay H100"

    def pactl(*args):  # LC_ALL=C: in comma-decimal locales pactl's JSON has 0,00 and doesn't parse
        return subprocess.run(["pactl", *args], capture_output=True, text=True, env={**os.environ, "LC_ALL": "C"}).stdout

    def dev(kind):
        name = pactl(f"get-default-{kind}").strip()
        found = pactl("-f", "json", "list", kind + "s")
        label = next((short(d) for d in json.loads(found or "[]") if d["name"] == name), name)
        return (name + ".monitor" if name and kind == "sink" else name), label
    return [dev("source"), dev("sink")]


def db(chunk):
    rms = audioop.rms(chunk, 2)
    return 20 * math.log10(rms / 32768) if rms else -90.0


def bar(f, n=10):  # f: 0..1
    k = round(max(0.0, min(1.0, f)) * n)
    return "▮" * k + "▯" * (n - k)


GPU = {"now": None}  # latest nvidia-smi reading, refreshed by watch_gpu()
HIST = collections.deque([0.0] * 30, maxlen=30)  # GPU % per second


def watch_gpu():
    q = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    while True:
        try:
            r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5)
            num = lambda x: float(x) if x.strip().replace(".", "").isdigit() else 0.0  # "[N/A]" -> 0
            now = [num(x) for x in r.stdout.splitlines()[0].split(",")]
            # errors ("No devices were found", NVML after a driver update) come on stdout too
            GPU["now"] = now if r.returncode == 0 and len(now) == 5 else None
            if GPU["now"]:
                HIST.append(now[0])
        except (OSError, IndexError, subprocess.TimeoutExpired):
            GPU["now"] = None
        time.sleep(1)


def gpu_lines():
    if not GPU["now"]:
        return ["gpu    n/a"]
    util, used, total, temp, watts = GPU["now"]
    spark = "".join("▁▂▃▄▅▆▇█"[min(7, int(u / 12.5))] for u in HIST)
    return [f"gpu    {bar(util / 100)}   {util:3.0f} %   {temp:.0f}°C   {watts:.0f} W",
            f"vram   {bar(used / max(total, 1))}   {used / 1024:.1f} / {total / 1024:.1f} GB",
            f"       {spark}  last 30 s"]


def draw(lines, w=W):
    if hup:  # window is gone
        return
    out = ["\x1b[H", BG, FG, "╭─ meeting-rec " + "─" * (w - 16) + "╮\x1b[K"]
    out += ["\r\n│ " + l[:w - 4].ljust(w - 4) + " │\x1b[K" for l in lines]
    out.append("\r\n╰" + "─" * (w - 2) + "╯\x1b[J")
    try:
        sys.stdout.write("".join(out)); sys.stdout.flush()
    except OSError:
        pass


def clock(secs):
    return f"{int(secs // 3600):02d}:{int(secs % 3600 // 60):02d}:{int(secs % 60):02d}"


def key():
    if hup or not select.select([0], [], [], 0)[0]:
        return None
    try:
        return os.read(0, 1)
    except OSError:
        return None


def record(path):
    devs = devices()
    streams = [parec(src) for src, _ in devs]  # mic, output
    wf = wave.open(path, "wb")
    wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(RATE)
    frames, silent, names_at, muted = 0, 0.0, time.time(), False
    try:
        while not stop:
            a, b = (s.stdout.read(BLOCK) for s in streams)
            if len(a) < BLOCK or len(b) < BLOCK:
                break  # a capture stream died; keep what we have
            if muted:
                a = bytes(BLOCK)  # silence keeps the timeline; the meeting app still hears the mic
            wf.writeframes(audioop.add(audioop.tostereo(a, 2, 1, 0), audioop.tostereo(b, 2, 0, 1), 2))
            frames += BLOCK // 2
            da, dbb = db(a), db(b)
            silent = silent + 0.1 if da < -55 and not muted else 0.0
            if time.time() - names_at > 3:
                names_at = time.time()
                for i, (src, label) in enumerate(devices()):
                    # default changed, e.g. headset plugged in: follow it, like the meeting app does
                    # shortcut: the switch leaves a ~0.2 s gap in that channel, fine for speech
                    if src and src != devs[i][0]:
                        streams[i].kill(); streams[i] = parec(src)
                    devs[i] = (src or devs[i][0], label)
            draw([f"● REC   {clock(frames / RATE)}", "",
                  f"mic    {devs[0][1][:18]:<18} " + (f"{bar(0)}  muted" if muted else
                                                         f"{bar((da + 60) / 60)}  {da:4.0f} dB"),
                  f"audio  {devs[1][1][:18]:<18} {bar((dbb + 60) / 60)}  {dbb:4.0f} dB", "",
                  f"⚠ mic silent for {silent:.0f} s" if silent >= 30 else "", ""]
                 + gpu_lines() + ["", f"[enter] stop & process   [m] {'unmute' if muted else 'mute'} mic"])
            k = key()
            if k in (b"\n", b"\r"):
                break
            muted ^= k == b"m"
    finally:
        for s in streams:
            s.kill()
        wf.close()
    return frames / RATE


def tail(path, n=10):
    """Last n non-empty lines of a step log, progress bars reduced to their latest state."""
    try:
        text = open(path, errors="replace").read()[-20000:]
    except FileNotFoundError:
        return []
    return [l for l in (x.split("\r")[-1].rstrip() for x in text.splitlines()) if l][-n:]


def progress(path):
    """Latest 'progress: …' line the step script printed (e.g. 'piece 4/11 · 9–12 min')."""
    return next((l[10:] for l in reversed(tail(path, 200)) if l.startswith("progress: ")), "")


def follow(name, secs):
    """Show process progress until the user closes the window (processing keeps going).
    [l] switches to the live log of the current step."""
    run, steps = f"{HERE}/runs/{name}", ["language", "words", "speakers", "summary", "note"]
    note, current, show_log = None, None, False
    while not hup:
        state, waiting = {}, False
        try:
            for line in open(f"{run}/progress.log"):
                p = line.split(maxsplit=3)
                if p[0] == "wait":  # another meeting is using the GPU
                    waiting = True
                elif p[0] == "start":
                    current, waiting = p[1], False
                    began = int(p[2]) if len(p) > 2 else time.time()
                    prog = progress(f"{run}/{p[1]}.log")
                    state[p[1]] = f"… {p[1]:<10} {prog[:26]:<26} {clock(time.time() - began)[3:]}"
                elif p[0] == "done" and p[1] in steps:
                    detail = p[3].strip() if len(p) > 3 else ""
                    state[p[1]] = f"✓ {p[1]:<10} {detail:<13} {p[2]} s"
                elif p[0] == "fail":
                    state[p[1]] = f"✗ {p[1]} failed   [l] see log"
                elif p[0] == "note":
                    note = line[5:].strip()
        except FileNotFoundError:
            pass
        head = [f"■ {clock(secs)} saved", "… waiting for the previous meeting" if waiting else ""]
        if show_log and current:
            cols = min(max(W, os.get_terminal_size().columns - 1), 140)
            log = [l.removeprefix("progress: ") for l in tail(f"{run}/{current}.log")]
            draw(head + [f"log: {current}", ""] + log + ["", "[l] steps   [q] close"], cols)
        else:
            lines = head + [state.get(s, "  " + s) for s in steps] + [""] + gpu_lines() + [""]
            lines.append(f"✓ Meetings/{name}.md  [o] open" if note else "")
            lines.append("[l] log   [q] close (processing continues)")
            draw(lines)
        k = key()
        if k == b"o" and note:
            # own session: else Obsidian, if this starts it, quits when the window closes
            subprocess.Popen(["xdg-open", "obsidian://open?path=" + urllib.parse.quote(note)], start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if k == b"l":
            show_log = not show_log
            sys.stdout.write("\x1b[2J")  # the two views have different widths
        if k in (b"q", b"\n", b"\r"):
            return
        time.sleep(0.3)


def release_pidfile():
    """Remove the pidfile only if it is still ours: a newer recorder may have written its own."""
    try:
        if open(PIDFILE).read() == str(os.getpid()):
            os.remove(PIDFILE)
    except OSError:
        pass


def main():
    name = datetime.now().strftime("%Y-%m-%d-%H%M")
    path = f"{HERE}/recordings/{name}.wav"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(s, on_signal)
    open(PIDFILE, "w").write(str(os.getpid()))
    threading.Thread(target=watch_gpu, daemon=True).start()
    sys.stdout.write(f"\x1b]0;meeting-rec\x07\x1b[?25l{BG}\x1b[2J")  # title, hide cursor, black
    old = termios.tcgetattr(0)
    tty.setcbreak(0)
    try:
        secs = record(path)
        release_pidfile()  # from here the hotkey starts a new recording
        # own session: processing survives this window closing
        subprocess.Popen([f"{HERE}/process", path], stdin=subprocess.DEVNULL,  # else ffmpeg eats keypresses
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        follow(name, secs)
    finally:
        release_pidfile()
        if not hup:
            termios.tcsetattr(0, termios.TCSADRAIN, old)


if __name__ == "__main__":
    main()
