"""
Tunnel helper — tries Cloudflare Quick Tunnel first, falls back to ngrok.

Cloudflare Quick Tunnel: free, no account, no 1-tunnel limit.
ngrok: fallback, requires NGROK_TOKEN env var.
"""
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request


def _cloudflared_bin() -> str:
    """Download cloudflared binary if needed, return path."""
    import stat
    dest = "/tmp/cloudflared"
    if not os.path.exists(dest):
        print("[tunnel] downloading cloudflared ...", flush=True)
        url = ("https://github.com/cloudflare/cloudflared/releases/latest"
               "/download/cloudflared-linux-amd64")
        urllib.request.urlretrieve(url, dest)
        os.chmod(dest, os.stat(dest).st_mode | stat.S_IEXEC)
    return dest


def _start_cloudflare(port: int) -> str:
    """Start cloudflared quick tunnel, return public HTTPS URL."""
    bin_path = _cloudflared_bin()
    proc = subprocess.Popen(
        [bin_path, "tunnel", "--url", f"http://localhost:{port}",
         "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    # cloudflared prints the URL to stderr/stdout within ~5s
    url = None
    deadline = time.time() + 30
    for raw in proc.stdout:
        line = raw.decode(errors="ignore")
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if m:
            url = m.group(0)
            break
        if time.time() > deadline:
            break
    if not url:
        proc.terminate()
        raise RuntimeError("cloudflared did not print a URL within 30s")
    # keep process alive in the background
    threading.Thread(target=proc.wait, daemon=True).start()
    return url


def _start_ngrok(port: int, token: str) -> str:
    """Start ngrok tunnel, return public HTTPS URL."""
    from pyngrok import ngrok
    ngrok.set_auth_token(token)
    try:
        for t in ngrok.get_tunnels():
            ngrok.disconnect(t.public_url)
    except Exception:
        pass
    tunnel = ngrok.connect(port, "http")
    return tunnel.public_url.replace("http://", "https://")


def start(port: int = 8000, token: str = None) -> str:
    """
    Open a public HTTPS tunnel on `port`.
    Tries Cloudflare Quick Tunnel first (no account needed).
    Falls back to ngrok if cloudflared fails.
    Returns the public URL.
    """
    # ── try Cloudflare Quick Tunnel ───────────────────────────────────────────
    try:
        url = _start_cloudflare(port)
        print(f"[tunnel] cloudflare quick tunnel: {url}", flush=True)
    except Exception as cf_err:
        print(f"[tunnel] cloudflare failed ({cf_err}), trying ngrok ...", flush=True)
        token = token or os.environ.get("NGROK_TOKEN") or os.environ.get("NGROK_AUTHTOKEN")
        if not token:
            raise RuntimeError(
                "No tunnel available. cloudflared failed and NGROK_TOKEN not set."
            )
        url = _start_ngrok(port, token)
        print(f"[tunnel] ngrok: {url}", flush=True)

    print(f"\n{'='*60}")
    print(f"  BADMINTON SERVER LIVE")
    print(f"  Public URL : {url}")
    print(f"  Camera     : POST {url}/frame")
    print(f"  Display WS : {url.replace('https','wss')}/ws")
    print(f"  Status     : {url}/status")
    print(f"{'='*60}\n")
    return url
