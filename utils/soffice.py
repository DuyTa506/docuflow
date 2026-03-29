"""
LibreOffice (soffice) runner with sandbox compatibility.

Handles environments where AF_UNIX sockets may be blocked (e.g. sandboxed
Linux VMs / containers) by detecting the restriction at runtime and applying
an LD_PRELOAD shim compiled on first use.

On Windows the shim is skipped entirely (AF_UNIX and gcc are not available).

Usage:
    from utils.soffice import run_soffice, get_soffice_env

    # Option 1 – run soffice directly
    result = run_soffice(["--headless", "--convert-to", "pdf", "input.docx"])

    # Option 2 – get env dict for your own subprocess call
    env = get_soffice_env()
    subprocess.run(["soffice", ...], env=env)
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def get_soffice_env() -> dict:
    """
    Return an environment dict suitable for running LibreOffice headless.

    On Linux/Mac: sets SAL_USE_VCLPLUGIN=svp and injects an LD_PRELOAD shim
    if AF_UNIX sockets are blocked.
    On Windows: returns a plain copy of os.environ (no shim needed/possible).
    """
    env = os.environ.copy()

    if sys.platform == "win32":
        return env  # No shim needed on Windows

    env["SAL_USE_VCLPLUGIN"] = "svp"

    if _needs_shim():
        shim = _ensure_shim()
        env["LD_PRELOAD"] = str(shim)

    return env


def run_soffice(args: list, **kwargs) -> subprocess.CompletedProcess:
    """
    Run LibreOffice with the appropriate environment.

    Args:
        args: Arguments to pass to soffice (e.g. ["--headless", "--convert-to", "docx", "file.doc"])
        **kwargs: Additional kwargs forwarded to subprocess.run()

    Returns:
        subprocess.CompletedProcess
    """
    env = get_soffice_env()
    return subprocess.run(["soffice"] + args, env=env, **kwargs)


# ── Shim internals (Linux/Mac only) ─────────────────────────────────────────

_SHIM_SO = Path(tempfile.gettempdir()) / "lo_socket_shim.so"


def _needs_shim() -> bool:
    """Return True if AF_UNIX sockets are blocked in this environment."""
    if sys.platform == "win32":
        return False
    try:
        import socket
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.close()
        return False
    except OSError:
        return True


def _ensure_shim() -> Path:
    """Compile the LD_PRELOAD shim if not already built, return its path."""
    if _SHIM_SO.exists():
        return _SHIM_SO

    src = Path(tempfile.gettempdir()) / "lo_socket_shim.c"
    src.write_text(_SHIM_SOURCE)
    subprocess.run(
        ["gcc", "-shared", "-fPIC", "-o", str(_SHIM_SO), str(src), "-ldl"],
        check=True,
        capture_output=True,
    )
    src.unlink()
    return _SHIM_SO


_SHIM_SOURCE = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>

static int (*real_socket)(int, int, int);
static int (*real_socketpair)(int, int, int, int[2]);
static int (*real_listen)(int, int);
static int (*real_accept)(int, struct sockaddr *, socklen_t *);
static int (*real_close)(int);
static int (*real_read)(int, void *, size_t);

/* Per-FD bookkeeping (FDs >= 1024 are passed through unshimmed). */
static int is_shimmed[1024];
static int peer_of[1024];
static int wake_r[1024];
static int wake_w[1024];
static int listener_fd = -1;

__attribute__((constructor))
static void init(void) {
    real_socket     = dlsym(RTLD_NEXT, "socket");
    real_socketpair = dlsym(RTLD_NEXT, "socketpair");
    real_listen     = dlsym(RTLD_NEXT, "listen");
    real_accept     = dlsym(RTLD_NEXT, "accept");
    real_close      = dlsym(RTLD_NEXT, "close");
    real_read       = dlsym(RTLD_NEXT, "read");
    for (int i = 0; i < 1024; i++) {
        peer_of[i] = -1;
        wake_r[i]  = -1;
        wake_w[i]  = -1;
    }
}

int socket(int domain, int type, int protocol) {
    if (domain == AF_UNIX) {
        int fd = real_socket(domain, type, protocol);
        if (fd >= 0) return fd;
        int sv[2];
        if (real_socketpair(domain, type, protocol, sv) == 0) {
            if (sv[0] >= 0 && sv[0] < 1024) {
                is_shimmed[sv[0]] = 1;
                peer_of[sv[0]]    = sv[1];
                int wp[2];
                if (pipe(wp) == 0) {
                    wake_r[sv[0]] = wp[0];
                    wake_w[sv[0]] = wp[1];
                }
            }
            return sv[0];
        }
        errno = EPERM;
        return -1;
    }
    return real_socket(domain, type, protocol);
}

int listen(int sockfd, int backlog) {
    if (sockfd >= 0 && sockfd < 1024 && is_shimmed[sockfd]) {
        listener_fd = sockfd;
        return 0;
    }
    return real_listen(sockfd, backlog);
}

int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen) {
    if (sockfd >= 0 && sockfd < 1024 && is_shimmed[sockfd]) {
        if (wake_r[sockfd] >= 0) {
            char buf;
            real_read(wake_r[sockfd], &buf, 1);
        }
        errno = ECONNABORTED;
        return -1;
    }
    return real_accept(sockfd, addr, addrlen);
}

int close(int fd) {
    if (fd >= 0 && fd < 1024 && is_shimmed[fd]) {
        int was_listener = (fd == listener_fd);
        is_shimmed[fd] = 0;
        if (wake_w[fd] >= 0) {
            char c = 0;
            write(wake_w[fd], &c, 1);
            real_close(wake_w[fd]);
            wake_w[fd] = -1;
        }
        if (wake_r[fd] >= 0) { real_close(wake_r[fd]); wake_r[fd] = -1; }
        if (peer_of[fd] >= 0) { real_close(peer_of[fd]); peer_of[fd] = -1; }
        if (was_listener) _exit(0);
    }
    return real_close(fd);
}
"""
