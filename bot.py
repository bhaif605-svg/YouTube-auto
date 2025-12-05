# bot.py
import os
import logging
import tempfile
import json
import re
import asyncio
from moviepy.editor import VideoFileClip
import yt_dlp
import openai
from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# Optional small web server
from aiohttp import web

# ---------------- CONFIG ----------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # change if you prefer another
# Optional: limit yt-dlp download duration in seconds (set to 60-90 for Shorts)
YTDLP_MAX_DURATION = int(os.getenv("YTDLP_MAX_DURATION", "90"))
# ----------------------------------------

openai.api_key = OPENAI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

YOUTUBE_REGEX = re.compile(r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/.*")

# ---------- Telegram handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Shorts Analyzer Bot ready.\nSend a YouTube Shorts link or upload a short video (<= 50MB)."
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send a YouTube Shorts link or upload a video file. I'll analyze and return metadata.")

def download_youtube(url: str, out_dir: str) -> str:
    # yt-dlp options — we limit duration to avoid huge downloads
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "merge_output_format": "mp4",
        "postprocessors": [],
        # restrict file duration (fallback) - yt-dlp has no simple max-duration filter here,
        # so we'll rely on downloads being short; yt-dlp supports --download-sections in CLI if needed.
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        file_path = ydl.prepare_filename(info)
        # handle typical extensions
        if os.path.exists(file_path):
            return file_path
        base = os.path.join(out_dir, info.get("id"))
        for ext in ("mp4","mkv","webm","m4a","mp3"):
            p = f"{base}.{ext}"
            if os.path.exists(p):
                return p
    raise FileNotFoundError("Downloaded file not found")

def extract_audio(video_path: str, audio_out: str):
    clip = VideoFileClip(video_path)
    # write audio — moviepy uses ffmpeg from the environment
    clip.audio.write_audiofile(audio_out, logger=None)
    clip.close()

def transcribe_with_openai(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        resp = openai.Audio.transcribe(WHISPER_MODEL, f)
        return resp.get("text", "")

def ask_llm_for_metadata(transcript: str, video_meta: dict) -> dict:
    system_prompt = (
        "You are a YouTube Shorts optimization assistant. Given transcript and metadata, "
        "produce a short viral title (<=70 chars), an SEO-optimized description (2 short paragraphs), "
        "up to 12 hashtags prefixed with # (comma separated), and 8-12 trending keywords. "
        "Return strictly JSON with keys: title, description, hashtags, keywords."
    )
    user_prompt = f"Transcript:\n{transcript[:4000]}\n\nMetadata:\n{json.dumps(video_meta)}\n\nReturn strictly JSON."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    resp = openai.ChatCompletion.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=600,
        temperature=0.7
    )
    text = resp["choices"][0]["message"]["content"].strip()
    # extract JSON object from text
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        json_text = text[start:end]
        return json.loads(json_text)
    except Exception:
        return {"title": "", "description": text, "hashtags": "", "keywords": []}

async def process_video_file(update: Update, video_path: str, source_url: str = None):
    status = await update.message.reply_text("Processing video — this may take 1-3 minutes...")
    tmpdir = tempfile.mkdtemp(prefix="shorts_")
    try:
        audio_path = os.path.join(tmpdir, "audio.wav")
        extract_audio(video_path, audio_path)
        await status.edit_text("Transcribing audio (OpenAI Whisper)...")
        transcript = transcribe_with_openai(audio_path)
        await status.edit_text("Generating Title/Description/Hashtags...")
        video_meta = {"source_url": source_url, "uploader": update.effective_user.username or str(update.effective_user.id)}
        metadata = ask_llm_for_metadata(transcript, video_meta)
        title = metadata.get("title", "").strip()
        description = metadata.get("description", "").strip()
        hashtags = metadata.get("hashtags", "")
        keywords = metadata.get("keywords", [])
        reply = f"*Viral Title:*\n{title}\n\n*Description:*\n{description}\n\n*Hashtags:*\n{hashtags}\n\n*Keywords:*\n{', '.join(keywords)}"
        await update.message.reply_text(reply, parse_mode="Markdown")
        # send JSON file
        json_path = os.path.join(tmpdir, "result.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(metadata, jf, ensure_ascii=False, indent=2)
        await update.message.reply_document(document=InputFile(json_path), filename="shorts_metadata.json")
        await status.delete()
    except Exception as e:
        logger.exception("Processing error")
        try:
            await status.edit_text(f"Processing failed: {e}")
        except:
            pass
    finally:
        try:
            for f in os.listdir(tmpdir):
                os.remove(os.path.join(tmpdir, f))
            os.rmdir(tmpdir)
        except Exception:
            pass

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    m = update.message
    text = (m.text or "").strip()
    if text and YOUTUBE_REGEX.search(text):
        await update.message.reply_text("YouTube link detected — downloading...")
        tmpdir = tempfile.mkdtemp(prefix="dl_")
        try:
            video_path = download_youtube(text, tmpdir)
            await process_video_file(update, video_path, source_url=text)
        except Exception as e:
            logger.exception("download error")
            await update.message.reply_text(f"Download failed: {e}")
        finally:
            try:
                for f in os.listdir(tmpdir):
                    os.remove(os.path.join(tmpdir, f))
                os.rmdir(tmpdir)
            except Exception:
                pass
        return

    if m.video or m.document:
        await update.message.reply_text("Video file received — saving...")
        tmpdir = tempfile.mkdtemp(prefix="up_")
        try:
            file = m.video or m.document
            file_obj = await file.get_file()
            local_path = os.path.join(tmpdir, file.file_name or f"{file.file_id}.mp4")
            await file_obj.download_to_drive(custom_path=local_path)
            await process_video_file(update, local_path)
        except Exception as e:
            logger.exception("upload error")
            await update.message.reply_text(f"Upload failed: {e}")
        finally:
            try:
                for f in os.listdir(tmpdir):
                    os.remove(os.path.join(tmpdir, f))
                os.rmdir(tmpdir)
            except Exception:
                pass
        return

    await update.message.reply_text("Send a YouTube Shorts link or upload a short video file.")

# ---------- Small keep-alive web server (optional) ----------
async def handle_root(request):
    return web.Response(text="OK")

def start_webserver():
    app = web.Application()
    app.router.add_get("/", handle_root)
    port = int(os.environ.get("PORT", "3000"))
    runner = web.AppRunner(app)
    async def _run():
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", port)
        await site.start()
    return _run

# ---------- Main ----------
def main():
    if not TELEGRAM_BOT_TOKEN or not OPENAI_API_KEY:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN and OPENAI_API_KEY in environment variables (Replit Secrets).")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.ALL & (~filters.COMMAND), handle_message))

    logger.info("Starting bot (polling)...")

    # run bot + optional webserver together using asyncio
    loop = asyncio.get_event_loop()
    web_runner = start_webserver()
    loop.create_task(web_runner())
    app.run_polling()

if __name__ == "__main__":
    main()
