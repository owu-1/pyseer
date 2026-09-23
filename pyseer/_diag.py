# -*- coding: utf-8 -*-
"""Optional timing and CPU-utilisation instrumentation for pyseer.

OFF unless PYSEER_DIAG=1. Nothing here changes results; it only measures.

    PYSEER_DIAG=1 pyseer --kmers ... --phenotypes ... --wg enet --cpu 10

Output is one line per phase on stderr:

    [pyseer-diag] phase                        wall     8.20s  cpu     8.30s  cores   1.01

`cores` is cpu-seconds / wall-seconds, so 1.00 means one thread was busy and
10.00 means ten were. A phase with cores ~1.0 that dominates wall time is work
`--cpu` cannot touch -- that is the whole point of this module.

Two extras:

* every glmnet() call in the parent process is logged with its matrix shape,
  which separates the serial full-data fit from the parallel fold fits
* a background thread samples CPU usage of the whole PROCESS GROUP once a
  second and, at exit, reports the mean/peak core count and the fraction of
  seconds spent below 1.5 and 3 cores -- i.e. directly answers "is pyseer
  actually using only one core?"

Set PYSEER_DIAG_TIMELINE to choose the timeline file (default
./pyseer_diag_<pid>.timeline.tsv).
"""

import atexit
import os
import sys
import threading
import time
from contextlib import contextmanager

_ENABLED = os.environ.get("PYSEER_DIAG", "").strip().lower() not in ("", "0", "false", "no")
ENABLED = _ENABLED

_T0 = time.time()
_TICKS = os.sysconf("SC_CLK_TCK")
_PGID = os.getpgrp()
_installed = False
_sampler = None


def _log(msg):
    sys.stderr.write("[pyseer-diag] " + msg + "\n")
    sys.stderr.flush()


def _group_cpu_seconds():
    """CPU seconds of every LIVE process in our process group (self + workers).

    Reads /proc directly rather than resource.getrusage(RUSAGE_SELF), because the
    joblib/loky fold workers are separate processes and their CPU time would
    otherwise be invisible.
    """
    total = 0.0
    try:
        entries = os.listdir("/proc")
    except OSError:
        return 0.0
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % entry) as fh:
                data = fh.read()
            fields = data[data.rindex(")") + 2:].split()
            if int(fields[2]) != _PGID:          # fields[2] == pgrp
                continue
            total += (int(fields[11]) + int(fields[12])) / _TICKS   # utime + stime
        except (OSError, ValueError, IndexError):
            continue
    return total


def cpu_seconds():
    """Live group CPU + CPU of already-reaped children (the loky workers)."""
    import resource
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return _group_cpu_seconds() + r.ru_utime + r.ru_stime


@contextmanager
def phase(label, extra=""):
    """Time a block, reporting wall, CPU and effective cores."""
    if not _ENABLED:
        yield
        return
    w0, c0 = time.time(), cpu_seconds()
    try:
        yield
    finally:
        wall = time.time() - w0
        cpu = cpu_seconds() - c0
        _log("%-34s wall %8.2fs  cpu %9.2fs  cores %6.2f %s"
             % (label, wall, cpu, (cpu / wall) if wall > 0 else 0.0, extra))


def log(msg):
    """Public helper: write one diagnostic line. No-op when disabled."""
    if _ENABLED:
        _log(msg)


def snapshot():
    """Take a timing checkpoint. Cheap and returns a dummy when disabled."""
    if not _ENABLED:
        return (0.0, 0.0)
    return (time.time(), cpu_seconds())


def since(label, snap, extra=""):
    """Log wall/cpu/cores elapsed since a snapshot().

    Used instead of the `phase` context manager where wrapping in a `with` would
    mean re-indenting large existing blocks.
    """
    if not _ENABLED:
        return
    w0, c0 = snap
    wall = time.time() - w0
    cpu = cpu_seconds() - c0
    _log("%-34s wall %8.2fs  cpu %9.2fs  cores %6.2f %s"
         % (label, wall, cpu, (cpu / wall) if wall > 0 else 0.0, extra))


class _CPUSampler(threading.Thread):
    """Sample process-group CPU once a second to build a core-count timeline."""

    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_flag = threading.Event()
        self.samples = []

    def run(self):
        last_t, last_c = time.time(), cpu_seconds()
        while not self.stop_flag.wait(self.interval):
            t, c = time.time(), cpu_seconds()
            dt, dc = t - last_t, c - last_c
            if dt > 0:
                self.samples.append((t - _T0, dc / dt, c))
            last_t, last_c = t, c

    def report(self):
        if not self.samples:
            _log("cpu sampler: no samples collected")
            return
        cores = [s[1] for s in self.samples]
        n = len(cores)
        _log("cpu timeline: %d samples over %.1fs" % (n, self.samples[-1][0]))
        _log("  mean cores %.2f   peak cores %.2f" % (sum(cores) / n, max(cores)))
        _log("  time below 1.5 cores: %5.1f%%"
             % (100 * sum(1 for c in cores if c < 1.5) / n))
        _log("  time below 3.0 cores: %5.1f%%"
             % (100 * sum(1 for c in cores if c < 3.0) / n))
        path = os.environ.get("PYSEER_DIAG_TIMELINE",
                              "pyseer_diag_%d.timeline.tsv" % os.getpid())
        try:
            with open(path, "w") as fh:
                fh.write("elapsed_s\tcores\tcumulative_cpu_s\n")
                for t, c, cum in self.samples:
                    fh.write("%.2f\t%.3f\t%.3f\n" % (t, c, cum))
            _log("  timeline written to %s" % path)
        except OSError as exc:
            _log("  could not write timeline: %s" % exc)


def _wrap_glmnet(fn):
    def wrapper(*args, **kwargs):
        x = kwargs.get("x", args[0] if args else None)
        shape = getattr(x, "shape", "?")
        w0, c0 = time.time(), cpu_seconds()
        out = fn(*args, **kwargs)
        wall = time.time() - w0
        cpu = cpu_seconds() - c0
        _log("  glmnet() shape=%-16s wall %8.2fs  cores %5.2f"
             % (shape, wall, (cpu / wall) if wall > 0 else 0.0))
        return out
    wrapper.__name__ = getattr(fn, "__name__", "glmnet")
    return wrapper


def _patch_glmnet():
    """Time glmnet() calls made from the parent, incl. cvglmnet's full-data fit.

    Fold fits run inside loky workers and will NOT be patched -- that is fine,
    they are covered by the enclosing cvglmnet phase duration.
    """
    try:
        import cvglmnet as cvmod
    except ImportError as exc:
        _log("could not import cvglmnet, glmnet timings unavailable: %s" % exc)
        return
    if hasattr(cvmod, "glmnet"):
        cvmod.glmnet = _wrap_glmnet(cvmod.glmnet)
        _log("patched cvglmnet.glmnet (full-data fit timing)")
    else:
        _log("cvglmnet has no 'glmnet' attribute; full-data fit not timed")


def _report_env():
    import multiprocessing
    _log("=" * 72)
    _log("pyseer diagnostics enabled (PYSEER_DIAG=%s)" % os.environ["PYSEER_DIAG"])
    _log("  pid %d   process group %d   cpu_count %d"
         % (os.getpid(), _PGID, multiprocessing.cpu_count()))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "TMPDIR", "JOBLIB_TEMP_FOLDER"):
        _log("  %-22s = %s" % (var, os.environ.get(var, "<unset>")))
    try:
        import joblib
        from joblib import parallel
        _log("  joblib %s   backend=%s" % (joblib.__version__,
                                           parallel.get_active_backend()[0]))
        _log("  joblib temp dir = %s" % parallel.get_temp_dir())
    except Exception as exc:        # noqa: BLE001 - diagnostics must never fail
        _log("  joblib introspection failed: %s" % exc)
    try:
        import glmnet_python
        _log("  glmnet_python from %s" % os.path.dirname(glmnet_python.__file__))
    except Exception as exc:        # noqa: BLE001
        _log("  glmnet_python introspection failed: %s" % exc)
    _log("=" * 72)


def install():
    """Idempotent. Call once, early in main()."""
    global _installed, _sampler
    if not _ENABLED or _installed:
        return
    _installed = True
    _report_env()
    _patch_glmnet()
    _sampler = _CPUSampler()
    _sampler.start()

    def _finish():
        try:
            _sampler.stop_flag.set()
            time.sleep(0.05)
            _log("total wall %.2fs   total cpu %.2fs"
                 % (time.time() - _T0, cpu_seconds()))
            _sampler.report()
        except Exception:           # noqa: BLE001
            pass

    atexit.register(_finish)

