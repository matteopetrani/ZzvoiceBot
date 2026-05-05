# ZzvoiceBot

A Telegram bot that transcribes voice messages and audio files. Send it a voice note, get back the text.

Written in C, built on [botlib](https://github.com/antirez/botlib). Runs well on small ARM boards like Raspberry Pi 5.

## How it works

The bot downloads audio from Telegram, converts it to 16 kHz mono WAV via ffmpeg, then sends it to the configured ASR backend. The result is posted back as a Telegram message.

Requests are serialised with a mutex (ASR is CPU-heavy or API-rate-limited, parallel requests hurt throughput). A C11 atomic tracks queue depth; if too many requests pile up, users are told to retry later.

## Backends

The backend is selected at compile time — uncomment exactly one in `whisperbot.c`:

```c
/* #define USE_WHISPER    */   // local whisper.cpp binary
/* #define USE_QWEN3_ASR  */   // local Qwen3-ASR binary
#define USE_GROQ_API           // Groq cloud STT API (default)
```

### Groq (default)

Uses the [Groq](https://console.groq.com) API with `whisper-large-v3-turbo`. Fast, accurate, free tier available. Requires a `groq_apikey.txt` file (see Configuration).

### whisper.cpp / Qwen3-ASR

Local inference via an external binary. No cloud dependency, no API key. Requires building the binary and downloading models; set the paths in the corresponding `#ifdef` block in `whisperbot.c`. When using a local backend the bot streams partial transcription results back to Telegram as text arrives.

## Dependencies

```
gcc, make
libcurl4-openssl-dev
libsqlite3-dev
ffmpeg
```

On Debian / Raspberry Pi OS:

```bash
sudo apt install build-essential libcurl4-openssl-dev libsqlite3-dev ffmpeg
```

## Build

```bash
make
```

Produces the `whisperbot` binary.

## Configuration

### Telegram bot token

Create a bot with [@BotFather](https://t.me/botfather) and save the token in `apikey.txt`:

```
123456789:AAFxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Groq API key

Get a key at [console.groq.com](https://console.groq.com) and save it in `groq_apikey.txt`:

```
gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Both files are listed in `.gitignore` and must never be committed.

### Allowed user

The bot only responds to one hardcoded Telegram user ID. Set it before building:

```c
#define ALLOWED_USER_ID 123456789LL   // whisperbot.c, line ~24
```

Find your ID by messaging [@userinfobot](https://t.me/userinfobot).

### Other tunables (top of `whisperbot.c`)

```c
#define MAX_QUEUE    10    // reject requests when queue exceeds this
#define MAX_SECONDS  300   // reject audio longer than this (seconds)
#define TIMEOUT      600   // hard kill the ASR process after this many seconds
```

## Running

### Manual

```bash
./whisperbot [--verbose] [--debug]
```

### systemd (run at boot, restart on failure)

Create `/etc/systemd/system/whisperbot.service`:

```ini
[Unit]
Description=ZzvoiceBot Telegram Voice Transcription Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/path/to/whisperbot
ExecStart=/path/to/whisperbot/whisperbot
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable whisperbot
sudo systemctl start whisperbot
```

View logs:

```bash
sudo journalctl -u whisperbot -f
```

## Limitations

- Allowed user list is a single hardcoded ID (edit and recompile to change).
- Backend and model paths for local backends are hardcoded.
- No persistence: in-flight requests are lost on restart.
