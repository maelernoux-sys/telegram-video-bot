#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import logging
import asyncio
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from datetime import datetime

# Telegram
from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

# Transcription + vidéo
import whisper
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ColorClip, vfx

# ------------------------------
# LOGGING
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ------------------------------
# ENV VARIABLES
# ------------------------------
TOKEN = os.environ.get("TG_BOT_TOKEN")
if not TOKEN:
    logger.error("La variable d'environnement TG_BOT_TOKEN n'est pas définie.")
    sys.exit(1)

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
FONT_PATH = "fonts/Montserrat-ExtraBold.ttf"

EXECUTOR = ThreadPoolExecutor(max_workers=2)
file_counter = count(start=1)

# ------------------------------
# OUTPUT FOLDER
# ------------------------------
OUTPUT_FOLDER = "output_videos"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
logger.info("Dossier de sortie : %s", OUTPUT_FOLDER)

# ------------------------------
# Whisper
# ------------------------------
logger.info("Chargement du modèle Whisper...")
model = whisper.load_model(WHISPER_MODEL)
logger.info("Modèle Whisper chargé.")

# ------------------------------
# VIDEO PROCESSING (mot par mot avec surlignage complet)
# ------------------------------
def process_video_capcut(input_file: str, counter_value: int) -> str:
    """
    Version mot par mot : surlignage bleu prend tout le mot.
    """
    clip = VideoFileClip(input_file).fx(vfx.mirror_x)
    result = model.transcribe(input_file)
    segments = result.get("segments", [])

    txt_clips = []

    for seg in segments:
        words = seg.get("words", [])

        if not words:
            # fallback
            text_words = seg["text"].strip().split()
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", seg_start + 2)
            words = [{"start": seg_start + i*((seg_end-seg_start)/len(text_words)),
                      "end": seg_start + (i+1)*((seg_end-seg_start)/len(text_words)),
                      "word": w} for i, w in enumerate(text_words)]

        for w in words:
            word_start = w["start"]
            word_end = w["end"]
            duration = max(word_end - word_start, 0.01)

            # Texte blanc centré
            word_clip = TextClip(w["word"], font=FONT_PATH, fontsize=60, color="white", method="caption")
            txt_w, txt_h = word_clip.size
            word_clip = word_clip.set_start(word_start).set_duration(duration).set_position(("center","center"))

            # Surlignage bleu complet sur le mot
            blue_clip = ColorClip(size=(txt_w, txt_h), color=(0,0,255)).set_start(word_start).set_duration(duration).set_position(("center","center"))

            txt_clips.append(blue_clip)
            txt_clips.append(word_clip)

    final = CompositeVideoClip([clip, *txt_clips])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_path = os.path.join(OUTPUT_FOLDER, f"video_{counter_value}_{timestamp}.mp4")
    final.write_videofile(dest_path, fps=clip.fps, codec="libx264", audio_codec="aac", verbose=False, logger=None)

    return os.path.abspath(dest_path)

# ------------------------------
# TELEGRAM HANDLER
# ------------------------------
async def download_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return

    file_obj = None
    counter_value = next(file_counter)

    if update.message.video:
        file_obj = update.message.video
    elif update.message.document and update.message.document.mime_type == "video/mp4":
        file_obj = update.message.document
    else:
        return

    logger.info("Nouvelle vidéo : %s", file_obj.file_id)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        temp_input = tmp_in.name

    try:
        tele_file = await context.bot.get_file(file_obj.file_id)
        await tele_file.download_to_drive(temp_input)
        await update.message.reply_text("✅ Vidéo reçue — traitement en cours...")

        loop = asyncio.get_running_loop()
        final_path = await loop.run_in_executor(EXECUTOR, process_video_capcut, temp_input, counter_value)

        with open(final_path, "rb") as f:
            await context.bot.send_video(chat_id=update.effective_chat.id, video=InputFile(f),
                                         caption="✅ Traitement CapCut PRO terminé")

    except Exception as e:
        logger.exception(e)
        await update.message.reply_text(f"❌ Erreur : {e}")

    finally:
        if os.path.exists(temp_input):
            os.remove(temp_input)

# ------------------------------
# LANCEMENT BOT
# ------------------------------
def main():
    logger.info("Démarrage du bot Telegram...")
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.VIDEO | filters.Document.MimeType("video/mp4"), download_video))
    logger.info("🤖 Bot en écoute...")
    app.run_polling()

if __name__ == "__main__":
    main()
