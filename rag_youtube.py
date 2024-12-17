import os
import tempfile
import logging
from uuid import uuid4
import yt_dlp
from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    Message,
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    CallbackQuery,
)
from aiogram.exceptions import TelegramAPIError
from aiogram.utils.keyboard import InlineKeyboardBuilder
import whisper
import asyncio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from typing import Dict, Tuple
import signal
import warnings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN", "7833475402:AAGnOzurg8-_j6Saeo7wlr7lO0X04FgveXQ"
)
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")

user_sessions: Dict[int, dict] = {}
active_tasks: Dict[str, asyncio.Task] = {}
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)


model_name = "unsloth/gemma-2-9b-it-bnb-4bit"  # "unsloth/gemma-2-9b-bnb-4bit"
TOKENIZER = AutoTokenizer.from_pretrained(model_name)
MODEL = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")


modelPath = "sentence-transformers/all-MiniLM-l6-v2"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": False}

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name=modelPath, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)


def transcribe_audio_process(audio_path: str) -> str:
    try:
        logger.info(f"transcribe_audio_process started: {audio_path=}")
        model = whisper.load_model("small")
        result = model.transcribe(audio_path, language="en")
        logger.info(f"transcribe_audio_process finished: {audio_path=}")
        return result["text"]
    except Exception as e:
        logger.error(f"Error in transcription process: {e}")
        raise


async def download_audio(youtube_url: str) -> Tuple[str, str]:
    try:
        logger.info(f"download_audio started: {youtube_url=}")

        temp_dir = tempfile.gettempdir()

        ydl = yt_dlp.YoutubeDL(
            {
                "quiet": True,
                "verbose": False,
                "format": "bestaudio",
                "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
                "postprocessors": [
                    {
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                        "key": "FFmpegExtractAudio",
                    }
                ],
                "ffmpeg_location": os.path.realpath(
                    "C:\\Users\\Vladislav\\Downloads\\ffmpeg-7.1-essentials_build\\ffmpeg-7.1-essentials_build\\bin\\"
                ),
            }
        )

        info = ydl.extract_info(youtube_url, download=True)
        title = info.get("title", "Unknown Title")
        audio_path = os.path.join(temp_dir, f"{info['id']}.mp3")

        logger.info(f"download_audio finished: {youtube_url=}")
        return audio_path, title
    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        raise


def create_video_selection_keyboard(
    transcripts: Dict[str, Tuple[str, str]]
) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for video_id, (title, _) in transcripts.items():
        display_title = title[:30] + "..." if len(title) > 30 else title
        builder.button(text=display_title,
                       callback_data=f"select_video:{video_id}")
    builder.adjust(1)
    return builder.as_markup()


def get_llm_response(question, context):
    system_prompt = (
        "You are a highly specialized assistant. Your primary task is to answer questions "
        "based strictly on the context provided. Follow these rules:\n\n"
        "1. USE ONLY the information given in the provided context to answer the question.\n"
        "2. If the context does not contain the information needed to answer the question, "
        'explicitly RESPOND WITH: "I cannot answer the question based on the given context."\n'
        "3. DO NOT make assumptions, infer missing details, or include information not present in the context.\n"
        "4. Ensure that your answers are clear, concise, and directly address the question based on the context provided.\n\n"
    )

    prompt = f"{system_prompt}\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
    print(prompt)
    inputs = TOKENIZER(prompt, return_tensors="pt",
                       truncation=True, max_length=2048)

    outputs = MODEL.generate(
        inputs["input_ids"].to("cuda"),
        max_length=1000,
        num_return_sequences=1,
        do_sample=False,
    )
    answer = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    return answer.split("Answer:")[-1].strip()


async def answer_question(transcript: str, question: str) -> str:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=300)
    docs = text_splitter.create_documents([transcript])

    vector_store = FAISS.from_documents(docs, EMBEDDINGS)

    results = vector_store.similarity_search(
        question,
        k=3,
    )

    context = "\n".join([res.page_content for res in results])

    return f"Ответ по тексту на ваш вопрос:\n\n {get_llm_response(question, context)}"


bot = Bot(token=TELEGRAM_BOT_TOKEN)
router = Router()
process_pool = ProcessPoolExecutor(max_workers=MAX_WORKERS)


@router.callback_query(F.data.startswith("select_video:"))
async def handle_video_selection(callback: CallbackQuery):
    try:
        video_id = callback.data.split(":")[1]
        chat_id = callback.message.chat.id

        if (
            chat_id not in user_sessions
            or video_id not in user_sessions[chat_id]["transcripts"]
        ):
            await callback.answer("Видео не найдено. Попробуйте еще раз.")
            return

        user_sessions[chat_id]["selected_video"] = video_id
        title, _ = user_sessions[chat_id]["transcripts"][video_id]

        await callback.message.edit_text(
            f"✅ Выбрано видео: {title}\nТеперь вы можете задавать вопросы по этому видео.",
            reply_markup=None,
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
                    process_pool, transcribe_audio_process, audio_path
                )

                video_id = str(uuid4())
                user_sessions[chat_id]["transcripts"][video_id] = (
                    video_title,
                    transcript,
                )
                keyboard = create_video_selection_keyboard(
                    user_sessions[chat_id]["transcripts"]
                )

                await progress_msg.edit_text(
                    f"✅ Видео успешно обработано!\n\n"
                    f"📝 Название: {video_title}\n"
                    f"📊 Длина текста: {len(transcript)} символов\n"
                    f"🔄 Активных задач: {len(active_tasks)}/{MAX_WORKERS}\n\n"
                    "Выберите видео для работы:",
                    reply_markup=keyboard,
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
            keyboard = create_video_selection_keyboard(
                user_sessions[chat_id]["transcripts"]
            )

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
            reply_markup=keyboard,
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
            keyboard = create_video_selection_keyboard(
                user_sessions[chat_id]["transcripts"]
            )
            await message.answer(
                "📺 Пожалуйста, выберите видео для обработки вопроса:",
                reply_markup=keyboard,
            )
            return

        video_id = user_sessions[chat_id]["selected_video"]
        title, transcript = user_sessions[chat_id]["transcripts"][video_id]

        user_query = message.text
        answer = await answer_question(transcript, user_query)

        keyboard = create_video_selection_keyboard(
            user_sessions[chat_id]["transcripts"]
        )

        await message.answer(
            f"📝 Ваш вопрос по видео '{title}':\n{user_query}\n\n"
            f"{answer}\n\n"
            f"Выберите другое видео для вопросов:",
            reply_markup=keyboard,
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


async def shutdown(sig, loop):
    logger.info(f"Received exit signal {sig.name}...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    logger.info(f"Cancelling {len(tasks)} tasks...")
    [task.cancel() for task in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Tasks cancelled, stopping loop...")
    loop.stop()


async def main():
    dp = Dispatcher()
    dp.include_router(router)

    try:
        # Для корректного завершения на Windows
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        def stop():
            logger.info("Stopping bot...")
            stop_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda s, frame: stop())

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
