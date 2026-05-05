/* Copyright (c) 2026, Salvatore Sanfilippo <antirez at gmail dot com>
 * All rights reserved. BSD license. */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <stdatomic.h>
#include <stdarg.h>
#include <pthread.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <signal.h>

#include "botlib.h"

/* Configuration. */
#define MAX_QUEUE 10
#define MAX_SECONDS 300
#define MSG_LIMIT 4000
#define TIMEOUT 600
/* Select backend: uncomment exactly one. */
/* #define USE_WHISPER */
#define USE_QWEN3_ASR

#if defined(USE_WHISPER) && defined(USE_QWEN3_ASR)
#error "Define only one backend: USE_WHISPER or USE_QWEN3_ASR."
#elif !defined(USE_WHISPER) && !defined(USE_QWEN3_ASR)
#error "Define one backend: USE_WHISPER or USE_QWEN3_ASR."
#endif

#ifdef USE_WHISPER
#define MODEL_BACKEND_NAME "Whisper"
#define MODEL_BIN_PATH "/Users/antirez/hack/ai/whisper.cpp/build/bin/whisper-cli"
#define MODEL_FAST "/Users/antirez/hack/ai/whisper.cpp/models/ggml-base.bin"
#define MODEL_BEST "/Users/antirez/hack/ai/whisper.cpp/models/ggml-medium.bin"
#define MODEL_FAST_NAME "base"
#define MODEL_BEST_NAME "medium"
#define QUEUE_THRESHOLD_FAST 3 /* Use fast model when queue >= this */
#define SHORT_AUDIO_THRESHOLD 1.5
#define DEFAULT_LANG "it"
#endif

#ifdef USE_QWEN3_ASR
#define MODEL_BACKEND_NAME "Qwen3-ASR"
#define MODEL_BIN_PATH "/home/teone/qwen-asr/qwen_asr"
#define MODEL_FAST "/home/teone/qwen-asr/qwen3-asr-0.6b"
#define MODEL_BEST "/home/teone/qwen-asr/qwen3-asr-0.6b"
#define MODEL_FAST_NAME "0.6b"
#define MODEL_BEST_NAME "0.6b"
#define QUEUE_THRESHOLD_FAST 3 /* Use fast model when queue >= this */
#endif

#define EDIT_INTERVAL_MS 500    /* Min ms between message edits. */

/* Serialization: only one ASR process at a time. */
atomic_int QueueLen = 0;
pthread_mutex_t WhisperLock = PTHREAD_MUTEX_INITIALIZER;

/* Return current time in milliseconds. */
long long mstime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000LL + ts.tv_nsec / 1000000;
}

/* Run command with NULL-terminated args. Capture stdout in *out if not NULL.
 * Stderr goes to /dev/null. Returns 0 on success, -1 on error. */
int runCommand(sds *out, const char *cmd, ...) {
    /* Build argv from varargs. */
    va_list ap;
    const char *argv[64];
    int argc = 0;

    argv[argc++] = cmd;
    va_start(ap, cmd);
    while (argc < 63 && (argv[argc] = va_arg(ap, const char *)) != NULL)
        argc++;
    va_end(ap);
    argv[argc] = NULL;

    /* Create pipe if output requested. */
    int fd[2] = {-1, -1};
    if (out && pipe(fd) == -1) return -1;

    pid_t pid = fork();
    if (pid == -1) {
        if (out) {
            close(fd[0]);
            close(fd[1]);
        }
        return -1;
    }

    if (pid == 0) {
        if (out) {
            close(fd[0]);
            dup2(fd[1], STDOUT_FILENO);
            close(fd[1]);
        } else {
            int devnull = open("/dev/null", O_WRONLY);
            if (devnull != -1) { dup2(devnull, STDOUT_FILENO); close(devnull); }
        }
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull != -1) { dup2(devnull, STDERR_FILENO); close(devnull); }
        execvp(cmd, (char *const *)argv);
        _exit(1);
    }

    /* Parent: read output if requested. */
    if (out) {
        close(fd[1]);
        *out = sdsempty();
        char buf[1024];
        ssize_t n;
        while ((n = read(fd[0], buf, sizeof(buf)-1)) > 0) {
            buf[n] = '\0';
            *out = sdscat(*out, buf);
        }
        close(fd[0]);
    }

    int status;
    waitpid(pid, &status, 0);
    if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) {
        if (out) {
            sdsfree(*out);
            *out = NULL;
        }
        return -1;
    }
    return 0;
}

/* Return audio duration in seconds, or -1 on error. */
double getDuration(const char *path) {
    sds out;
    if (runCommand(&out, "ffprobe", "-i", path, "-show_entries",
                   "format=duration", "-v", "quiet", "-of", "csv=p=0",
                   NULL) != 0) return -1;
    double dur = atof(out);
    sdsfree(out);
    return dur;
}

/* Convert to 16khz mono WAV. */
int toWav(const char *in, const char *out, double duration) {
#ifdef USE_WHISPER
    if (duration < SHORT_AUDIO_THRESHOLD) {
        /* Pad short clips to improve whisper behavior. */
        char af[64];
        snprintf(af, sizeof(af), "apad=whole_dur=%.1f", SHORT_AUDIO_THRESHOLD);
        return runCommand(NULL, "ffmpeg", "-y", "-i", in, "-af", af,
                          "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                          out, NULL);
    }
#else
    UNUSED(duration);
#endif
    return runCommand(NULL, "ffmpeg", "-y", "-i", in,
                      "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                      out, NULL);
}

/* Run selected ASR backend with timeout. Streams output to Telegram by
 * editing 'msg_id'. Returns 0 on success, -1 on error. */
int transcribe(const char *wav, const char *model, int64_t target,
               int64_t chat_id, int64_t msg_id, int short_audio)
{
    int fd[2];
    if (pipe(fd) == -1) return -1;

    pid_t pid = fork();
    if (pid == -1) {
        close(fd[0]);
        close(fd[1]);
        return -1;
    }

    if (pid == 0) {
        close(fd[0]);
        dup2(fd[1], STDOUT_FILENO);
#ifdef USE_WHISPER
        dup2(fd[1], STDERR_FILENO);
        close(fd[1]);
        const char *lang = short_audio ? DEFAULT_LANG : "auto";
        execlp(MODEL_BIN_PATH, "whisper-cli",
               "-m", model,
               "-f", wav,
               "-l", lang,
               "-t", "20", "-np", "-nt", NULL);
#endif
#ifdef USE_QWEN3_ASR
        UNUSED(short_audio);
        close(fd[1]);
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull != -1) { dup2(devnull, STDERR_FILENO); close(devnull); }
        execlp(MODEL_BIN_PATH, "qwen_asr",
               "-d", model,
               "-i", wav,
               "-S", "20",
               NULL);
#endif
        _exit(1);
    }

    close(fd[1]);
    fcntl(fd[0], F_SETFL, O_NONBLOCK);

    sds text = sdsempty();
    time_t start = time(NULL);
    long long last_edit = 0;
    int status = 0;

    /* Read data as it is streamed by the selected ASR backend. */
    while (1) {
        /* Timeout check. */
        if (time(NULL) - start > TIMEOUT) {
            kill(pid, SIGKILL);
            waitpid(pid, NULL, 0);
            close(fd[0]);
            sdsfree(text);
            botEditMessageText(chat_id, msg_id, "Transcription timed out.");
            return -1;
        }

        /* Read available data. */
        char buf[1024];
        ssize_t n;
        while ((n = read(fd[0], buf, sizeof(buf)-1)) > 0) {
            buf[n] = '\0';
            text = sdscat(text, buf);
        }

        /* Message too long? Send and continue in new message. */
        if (sdslen(text) > MSG_LIMIT) {
            botEditMessageText(chat_id, msg_id, text);
            sdsfree(text);
            text = sdsnew("[...]\n");
            botSendMessageAndGetInfo(target, text, 0, &chat_id, &msg_id);
            last_edit = mstime();
        }

        /* Update message periodically. */
        if (sdslen(text) && mstime() - last_edit >= EDIT_INTERVAL_MS) {
            botEditMessageText(chat_id, msg_id, text);
            last_edit = mstime();
        }

        /* Child done? */
        if (waitpid(pid, &status, WNOHANG) == pid) {
            while ((n = read(fd[0], buf, sizeof(buf)-1)) > 0) {
                buf[n] = '\0';
                text = sdscat(text, buf);
            }
            break;
        }

        usleep(100000);
    }

    close(fd[0]);

    /* Check exit status. */
    int exit_ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;

    /* Trim whitespace from output. */
    sdstrim(text, " \t\r\n");

    /* Final update. */
    if (sdslen(text) > 0) {
        botEditMessageText(chat_id, msg_id, text);
    } else if (!exit_ok) {
        botEditMessageText(chat_id, msg_id, "Transcription failed.");
    } else {
        botEditMessageText(chat_id, msg_id, "(no speech detected)");
    }
    sdsfree(text);
    return exit_ok ? 0 : -1;
}

/* Check if file is audio based on mime type or extension. */
int isAudioFile(BotRequest *br) {
    const char *exts[] = {
        ".mp3", ".wav", ".ogg", ".oga", ".m4a", ".flac", ".opus",
        ".mpeg", ".mpga", ".wma", ".aac", ".webm", NULL
    };

    if (br->file_mime) {
        if (strstr(br->file_mime, "audio/")) return 1;
        if (strstr(br->file_mime, "ogg")) return 1;
    }
    if (br->file_name) {
        char *ext = strrchr(br->file_name, '.');
        if (ext) {
            for (int i = 0; exts[i]; i++)
                if (!strcasecmp(ext, exts[i])) return 1;
        }
    }
    return 0;
}

void handleRequest(sqlite3 *dbhandle, BotRequest *br) {
    UNUSED(dbhandle);

    int allowed = 0;
    for (int i = 0; i < NumAllowedUsers; i++)
        if (br->from == AllowedUserIDs[i]) { allowed = 1; break; }
    if (!allowed) return;

    /* Accept voice messages, audio files, or documents that look like audio. */
    int is_audio = 0;
    if (br->file_type == TB_FILE_TYPE_VOICE_OGG) is_audio = 1;
    if (br->file_type == TB_FILE_TYPE_AUDIO) is_audio = 1;
    if (br->file_type == TB_FILE_TYPE_DOCUMENT && isAudioFile(br)) is_audio = 1;
    if (!is_audio) return;

    /* Temp file names. Fixed extension - ffmpeg detects format from content,
     * otherwise we can expose the server to security issues because of
     * path traversal. */
    static atomic_int id = 0;
    int myid = atomic_fetch_add(&id, 1);
    char in[64], out[64];
    snprintf(in, sizeof(in), "/tmp/wb_%d_%d.audio", (int)getpid(), myid);
    snprintf(out, sizeof(out), "/tmp/wb_%d_%d.wav", (int)getpid(), myid);

    /* Download. */
    if (!botGetFile(br, in)) {
        botSendMessage(br->target, "Can't download audio.", br->msg_id);
        return;
    }

    /* Check duration. */
    double dur = getDuration(in);
    if (dur < 0 || dur > MAX_SECONDS) {
        char msg[128];
        if (dur < 0)
            snprintf(msg, sizeof(msg), "Can't read audio duration.");
        else
            snprintf(msg, sizeof(msg), "Audio too long: %.0fs (max %ds).",
                     dur, MAX_SECONDS);
        botSendMessage(br->target, msg, br->msg_id);
        unlink(in);
        return;
    }

    /* Convert. */
    if (toWav(in, out, dur) != 0) {
        botSendMessage(br->target, "Audio conversion failed.", br->msg_id);
        unlink(in);
        return;
    }
    unlink(in);

    /* Check queue. */
    int pos = atomic_fetch_add(&QueueLen, 1);
    if (pos >= MAX_QUEUE) {
        atomic_fetch_sub(&QueueLen, 1);
        botSendMessage(br->target, "Too busy, try later.", br->msg_id);
        unlink(out);
        return;
    }

    /* Notify user. */
    int64_t chat_id, msg_id;
    sds status = pos > 0
        ? sdscatprintf(sdsempty(), "Queued (%d)...", pos+1)
        : sdsnew("Transcribing...");
    botSendMessageAndGetInfo(br->target, status, br->msg_id, &chat_id, &msg_id);
    sdsfree(status);

    /* Wait for turn. */
    pthread_mutex_lock(&WhisperLock);

    /* Select model based on queue length. */
    int qlen = atomic_load(&QueueLen);
    const char *model = qlen >= QUEUE_THRESHOLD_FAST ? MODEL_FAST : MODEL_BEST;
    const char *mname = qlen >= QUEUE_THRESHOLD_FAST ? MODEL_FAST_NAME : MODEL_BEST_NAME;

    char msg[96];
    snprintf(msg, sizeof(msg), "Transcribing (%s %s)...",
             MODEL_BACKEND_NAME, mname);
    botEditMessageText(chat_id, msg_id, msg);

    /* Run the model, we pass the chat/msg ID since it will update
     * the message with actual transcription. */
#ifdef USE_WHISPER
    int short_audio = dur < SHORT_AUDIO_THRESHOLD;
#else
    int short_audio = 0;
#endif
    transcribe(out, model, br->target, chat_id, msg_id, short_audio);

    pthread_mutex_unlock(&WhisperLock);
    atomic_fetch_sub(&QueueLen, 1);
    unlink(out);
}

void cron(sqlite3 *dbhandle) {
    UNUSED(dbhandle);
}

int main(int argc, char **argv) {
    static char *triggers[] = {"*", NULL};
    printf("%s bot started. Queue max: %d, Audio max: %ds\n",
           MODEL_BACKEND_NAME, MAX_QUEUE, MAX_SECONDS);
    startBot(TB_CREATE_KV_STORE, argc, argv, TB_FLAGS_NONE,
             handleRequest, cron, triggers);
    return 0;
}
