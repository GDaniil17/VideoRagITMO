import os
import tempfile
import logging
from uuid import uuid4
import yt_dlp
from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from aiogram.exceptions import TelegramAPIError
from aiogram.utils.keyboard import InlineKeyboardBuilder
import whisper
import asyncio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from typing import Dict, Tuple
import signal
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7833475402:AAGnOzurg8-_j6Saeo7wlr7lO0X04FgveXQ")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")

user_sessions: Dict[int, dict] = {}
active_tasks: Dict[str, asyncio.Task] = {}
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)


def transcribe_audio_process(audio_path: str) -> str:
    try:
        logger.info(f"transcribe_audio_process started: {audio_path=}")
        model = whisper.load_model("small")
        result = model.transcribe(audio_path, language="ru")
        logger.info(f"transcribe_audio_process finished: {audio_path=}")
        return result["text"]
    except Exception as e:
        logger.error(f"Error in transcription process: {e}")
        raise


async def download_audio(youtube_url: str) -> Tuple[str, str]:
    try:
        logger.info(f"download_audio started: {youtube_url=}")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            audio_path = temp_file.name

        ydl_opts = {
            'format': '251',
            'outtmpl': audio_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': False,
            'noplaylist': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            title = info.get('title', 'Unknown Title')

        logger.info(f"download_audio finished: {youtube_url=}")
        return audio_path, title
    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        raise


def create_video_selection_keyboard(transcripts: Dict[str, Tuple[str, str]]) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for video_id, (title, _) in transcripts.items():
        display_title = title[:30] + "..." if len(title) > 30 else title
        builder.button(text=display_title, callback_data=f"select_video:{video_id}")
    builder.adjust(1)
    return builder.as_markup()


async def answer_question(transcript: str, question: str) -> str:
    return f"Ответ по тексту на ваш вопрос:\n\nКонтекст из транскрипции:\n{transcript[:500]}..."


bot = Bot(token=TELEGRAM_BOT_TOKEN)
router = Router()
process_pool = ProcessPoolExecutor(max_workers=MAX_WORKERS)


@router.callback_query(F.data.startswith("select_video:"))
async def handle_video_selection(callback: CallbackQuery):
    try:
        video_id = callback.data.split(":")[1]
        chat_id = callback.message.chat.id

        if chat_id not in user_sessions or video_id not in user_sessions[chat_id]["transcripts"]:
            await callback.answer("Видео не найдено. Попробуйте еще раз.")
            return

        user_sessions[chat_id]["selected_video"] = video_id
        title, _ = user_sessions[chat_id]["transcripts"][video_id]

        await callback.message.edit_text(
            f"✅ Выбрано видео: {title}\nТеперь вы можете задавать вопросы по этому видео.",
            reply_markup=None
        )
        await callback.answer("Видео выбрано!")
    except Exception as e:
        logger.error(f"Error in video selection handler: {e}")
        await callback.answer("Ошибка при выборе видео. Попробуйте еще раз.")


@router.message(Command("transcribe"))
async def transcribe_handler(message: Message):
    try:
        chat_id = message.chat.id
        if chat_id not in user_sessions:
            user_sessions[chat_id] = {"transcripts": {}}

        args = message.text.split(maxsplit=1)
        if len(args) < 2:
            await message.answer(
                "ℹ️ Использование: /transcribe <ссылка на YouTube>\nНапример: /transcribe https://youtube.com/watch?v=..."
            )
            return

        if len(active_tasks) >= MAX_WORKERS:
            await message.answer(
                "⚠️ Система сейчас загружена. Пожалуйста, подождите немного и попробуйте снова.\n"
                f"Активных задач: {len(active_tasks)}/{MAX_WORKERS}"
            )
            return

        youtube_url = args[1]
        progress_msg = await message.answer("⏳ Загрузка видео...")
        task_id = str(uuid4())

        async def process_video():
            audio_path = None
            try:
                audio_path, video_title = await download_audio(youtube_url)
                await progress_msg.edit_text("🔄 Преобразование речи в текст...")

                loop = asyncio.get_running_loop()
                transcript = await loop.run_in_executor(
                    process_pool,
                    transcribe_audio_process,
                    audio_path
                )

                video_id = str(uuid4())
                user_sessions[chat_id]["transcripts"][video_id] = (video_title, transcript)
                keyboard = create_video_selection_keyboard(user_sessions[chat_id]["transcripts"])

                await progress_msg.edit_text(
                    f"✅ Видео успешно обработано!\n\n"
                    f"📝 Название: {video_title}\n"
                    f"📊 Длина текста: {len(transcript)} символов\n"
                    f"🔄 Активных задач: {len(active_tasks)}/{MAX_WORKERS}\n\n"
                    "Выберите видео для работы:",
                    reply_markup=keyboard
                )
            except Exception as e:
                logger.error(f"Error processing video: {e}")
                await progress_msg.edit_text(f"❌ Ошибка при обработке видео: {str(e)}")
            finally:
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
                if task_id in active_tasks:
                    del active_tasks[task_id]

        task = asyncio.create_task(process_video())
        active_tasks[task_id] = task

    except Exception as e:
        logger.error(f"Error in transcribe handler: {e}")
        await message.answer(f"❌ Ошибка при обработке видео: {str(e)}")


@router.message(CommandStart())
async def start_handler(message: Message):
    try:
        chat_id = message.chat.id
        user_sessions[chat_id] = {"transcripts": {}}
        keyboard = None

        if user_sessions[chat_id]["transcripts"]:
            keyboard = create_video_selection_keyboard(user_sessions[chat_id]["transcripts"])

        await message.answer(
            "👋 Привет! Я бот для транскрибации YouTube видео.\n\n"
            "🎯 Что я умею:\n"
            "• Преобразовывать речь из видео в текст\n"
            "• Отвечать на вопросы по содержанию видео\n\n"
            "🚀 Как использовать:\n"
            "1. Отправьте команду /transcribe и ссылку на YouTube видео\n"
            "2. Дождитесь окончания обработки\n"
            "3. Выберите видео из списка\n"
            "4. Задавайте вопросы по содержанию!\n\n"
            f"ℹ️ Статус системы:\n"
            f"• Доступно процессов: {MAX_WORKERS}\n"
            f"• Активных задач: {len(active_tasks)}/{MAX_WORKERS}",
            reply_markup=keyboard
        )
    except TelegramAPIError as e:
        logger.error(f"Telegram API error in start handler: {e}")
        await message.answer("Произошла ошибка. Попробуйте позже.")


@router.message(F.text & ~F.command)
async def query_handler(message: Message):
    try:
        chat_id = message.chat.id
        if chat_id not in user_sessions or not user_sessions[chat_id]["transcripts"]:
            await message.answer(
                "❌ У вас нет загруженных видео.\n"
                "Используйте /transcribe <ссылка> для добавления видео."
            )
            return

        if "selected_video" not in user_sessions[chat_id]:
            keyboard = create_video_selection_keyboard(user_sessions[chat_id]["transcripts"])
            await message.answer(
                "📺 Пожалуйста, выберите видео для обработки вопроса:",
                reply_markup=keyboard
            )
            return

        video_id = user_sessions[chat_id]["selected_video"]
        title, transcript = user_sessions[chat_id]["transcripts"][video_id]

        user_query = message.text
        answer = await answer_question(transcript, user_query)

        keyboard = create_video_selection_keyboard(user_sessions[chat_id]["transcripts"])

        await message.answer(
            f"📝 Ваш вопрос по видео '{title}':\n{user_query}\n\n"
            f"{answer}\n\n"
            f"Выберите другое видео для вопросов:",
            reply_markup=keyboard
        )
    except TelegramAPIError as e:
        logger.error(f"Telegram API error in query handler: {e}")
        await message.answer("❌ Ошибка при обработке вашего вопроса.")


async def set_commands(bot: Bot):
    commands = [
        BotCommand(command="start", description="Запустить бота"),
        BotCommand(command="transcribe", description="Добавить YouTube видео"),
    ]
    await bot.set_my_commands(commands)


async def shutdown(signal, loop):
    logger.info(f"Received exit signal {signal.name}...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    logger.info(f"Cancelling {len(tasks)} outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)
    process_pool.shutdown(wait=True)
    loop.stop()


async def main():
    dp = Dispatcher()
    dp.include_router(router)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(shutdown(s, loop))
        )

    try:
        await set_commands(bot)
        logger.info(f"Starting bot with {MAX_WORKERS} workers...")
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
    finally:
        await bot.session.close()
        process_pool.shutdown(wait=True)


if __name__ == "__main__":
    asyncio.run(main())
