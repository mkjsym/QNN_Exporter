# Run a command with transparent huge pages disabled for the process tree (PR_SET_THP_DISABLE is inherited across fork and execve).
# Why: on 2026-10-19 the deep target export spent 99% of its CPU in failing kernel compaction (memory full, swap full), 93 s of user time in 4.3 h.
import ctypes, os, sys
libc = ctypes.CDLL(None, use_errno=True)
PR_SET_THP_DISABLE = 41
if libc.prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), 'prctl(PR_SET_THP_DISABLE) failed')
os.execvp(sys.argv[1], sys.argv[1:])
