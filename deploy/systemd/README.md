# systemd user units for the three llama-servers

These keep the servers the live test suite needs running across crashes and
reboots. They are the copies of what is installed at
`~/.config/systemd/user/`; edit there, or reinstall from here.

| Unit | Port | Model | Why its flags matter |
|---|---|---|---|
| `llama-chat.service` | 8090 | SmolLM2-360M-Instruct | `--metrics` makes `/metrics` real instead of 501 |
| `llama-embed.service` | 8081 | nomic-embed-text-v1.5 | `--embd-normalize -1` returns raw vectors; the default (L2) makes 4.1's ablation degenerate |
| `llama-rerank.service` | 8082 | bge-reranker-v2-m3 | `--reranking --pooling rank`; cannot share a process with embeddings |

`--reranking` forces pooling to `rank`, which corrupts `/v1/embeddings` on the
same server, so the reranker must stay a separate unit. See REAL_VS_MOCK.md §1a.

## Install

```bash
cp deploy/systemd/llama-*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now llama-chat llama-embed llama-rerank
loginctl enable-linger "$USER"   # start at boot without logging in
```

Without `enable-linger` the units start at **login**, not at boot.

## Operating

```bash
systemctl --user status llama-rerank
journalctl --user -u llama-rerank -n 50 -f
./start-servers.sh --status       # health + which model each serves
```

`start-servers.sh` detects these units and delegates to `systemctl`, so systemd
stays the single owner of each port. Without the units it launches the
processes directly.

`Restart=on-failure` with `RestartSec=5` covers crashes (verified: SIGKILL the
reranker and it returns). `TimeoutStartSec=900` allows a first run to download
a model via `-hf`.

Port 8080 on this machine is Open WebUI, not llama.cpp. Nothing here touches it.
