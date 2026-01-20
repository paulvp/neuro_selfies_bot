import logging
import os
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from cachetools import TTLCache
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
import boto3
from botocore.exceptions import ClientError
import random
import uuid
import urllib.parse
from yookassa import Configuration, Payment
import aiohttp
import json

from aiogram import Bot, Dispatcher, F, BaseMiddleware
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton,
    LabeledPrice, PreCheckoutQuery, InputMediaPhoto, FSInputFile
)
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.exceptions import TelegramBadRequest

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
BOT_USERNAME = os.getenv("BOT_USERNAME", "neuro_selfies_bot")
ADMIN_IDS = list(map(int, filter(None, os.getenv("ADMIN_USER_IDS", "").split(","))))

DATABASE_URL = os.getenv("DATABASE_URL")
db_pool = ThreadedConnectionPool(minconn=5, maxconn=50, dsn=DATABASE_URL)

R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL")

r2_client = boto3.client(
    's3',
    endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name='auto'
)

KIA_API_KEY = os.getenv("KIA_API_KEY", "")
KIA_API_URL = "https://api.kie.ai/api/v1/jobs"
KIA_MODEL = "google/nano-banana-edit"

YOOKASSA_SHOP_ID = os.getenv("YOOKASSA_SHOP_ID", "")
YOOKASSA_SECRET_KEY = os.getenv("YOOKASSA_SECRET_KEY", "")

if YOOKASSA_SHOP_ID and YOOKASSA_SECRET_KEY:
    Configuration.account_id = YOOKASSA_SHOP_ID
    Configuration.secret_key = YOOKASSA_SECRET_KEY
    yookassa_enabled = True
else:
    yookassa_enabled = False
    logger.warning("YooKassa not configured")

PACKAGE_10 = {"credits": 10, "price_rub": 69, "price_stars": 69}
PACKAGE_50 = {"credits": 50, "price_rub": 299, "price_stars": 299}
PACKAGE_100 = {"credits": 100, "price_rub": 499, "price_stars": 499}
REFERRAL_BONUS = 2

active_polls = set()

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

user_cache = TTLCache(maxsize=50000, ttl=300)

class RateLimitMiddleware(BaseMiddleware):
    """Silent rate limiting: 1 message per second"""
    def __init__(self):
        self.user_last_message = TTLCache(maxsize=50000, ttl=1)
    
    async def __call__(self, handler, event, data: Dict[str, Any]):
        user_id = getattr(event, 'from_user', None)
        if user_id:
            user_id = user_id.id
            if user_id in ADMIN_IDS:
                return await handler(event, data)
            
            if user_id in self.user_last_message:
                logger.warning(f"Rate limit: user {user_id} dropped")
                return
            
            self.user_last_message[user_id] = True
        
        return await handler(event, data)

class CallbackSecurityMiddleware(BaseMiddleware):
    """Validate callback queries to prevent hijacking"""
    async def __call__(self, handler, event: CallbackQuery, data: Dict[str, Any]):
        callback = event
        user_id = callback.from_user.id
        
        if callback.data and callback.data.startswith("admin"):
            if user_id not in ADMIN_IDS:
                await callback.answer("⛔️ Доступ запрещен: требуются права администратора.", show_alert=True)
                logger.warning(f"SECURITY: Unauthorized admin callback attempt by {user_id}")
                return
        
        user_specific_callbacks = [
            "pay_",           # Payment callbacks (pay_10_rub, pay_50_stars, etc)
            "retry_yookassa", # Retry payment
            "buy_credits",    # Buy credits menu
            "invite_friend"   # Referral system
        ]
        
        if callback.data:
            for specific_callback in user_specific_callbacks:
                if callback.data.startswith(specific_callback):
                    if callback.message and callback.message.chat.id != user_id:
                        await callback.answer("⛔️ Это не ваше сообщение.", show_alert=True)
                        logger.warning(f"SECURITY: User {user_id} tried to access another user's callback")
                        return
                    break
        
        return await handler(event, data)

class GenerationStates(StatesGroup):
    waiting_for_photo = State()
    choosing_celebrity = State()
    choosing_action = State()

def get_db_connection():
    return db_pool.getconn()

def return_db_connection(conn):
    db_pool.putconn(conn)

async def ensure_user_exists(user_id: int, username: str, referred_by: str = None):
    """Create user if not exists, give 2 free credits"""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
            if not cur.fetchone():
                referral_code = f"REF{user_id}"
                
                cur.execute("""
                    INSERT INTO users (user_id, username, credits, additional_credits, 
                                     spend_rub, spend_credits, referral_code, referred_by)
                    VALUES (%s, %s, 2, 0, 0, 0, %s, %s)
                """, (user_id, username, referral_code, referred_by))
                conn.commit()
                
                if referred_by:
                    cur.execute("""
                        UPDATE users 
                        SET additional_credits = additional_credits + %s,
                            referrals = referrals + 1
                        WHERE referral_code = %s
                    """, (REFERRAL_BONUS, referred_by))
                    conn.commit()
                
                logger.info(f"Created new user {user_id} with 2 free credits")
    finally:
        return_db_connection(conn)

async def get_user_data(user_id: int, use_cache: bool = True) -> Optional[Dict]:
    """Get user data from database with caching"""
    if use_cache and user_id in user_cache:
        return user_cache[user_id]
    
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            if row:
                user_data = dict(row)
                user_cache[user_id] = user_data
                return user_data
            return None
    finally:
        return_db_connection(conn)

async def deduct_credits(user_id: int, amount: int = 1) -> bool:
    """Deduct credits from user (regular first, then additional)"""
    user_cache.pop(user_id, None)
    
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT credits, additional_credits 
                FROM users WHERE user_id = %s FOR UPDATE
            """, (user_id,))
            user = cur.fetchone()
            
            if not user:
                return False
            
            total = user['credits'] + user['additional_credits']
            if total < amount:
                return False
            
            if user['credits'] >= amount:
                cur.execute("""
                    UPDATE users 
                    SET credits = credits - %s, spend_credits = spend_credits + %s
                    WHERE user_id = %s
                """, (amount, amount, user_id))
            else:
                remaining = amount - user['credits']
                cur.execute("""
                    UPDATE users 
                    SET credits = 0, 
                        additional_credits = additional_credits - %s,
                        spend_credits = spend_credits + %s
                    WHERE user_id = %s
                """, (remaining, amount, user_id))
            
            conn.commit()
            return True
    finally:
        return_db_connection(conn)

async def add_credits(user_id: int, amount: int, payment_type: str, payment_sum: float):
    """Add credits to user and record payment"""
    user_cache.pop(user_id, None)
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE users 
                SET credits = credits + %s, spend_rub = spend_rub + %s
                WHERE user_id = %s
            """, (amount, payment_sum, user_id))
            
            cur.execute("""
                INSERT INTO payments (user_id, amount, currency, type, payment_sum)
                VALUES (%s, %s, %s, %s, %s)
            """, (user_id, amount, payment_type, 'credits', payment_sum))
            
            conn.commit()
    finally:
        return_db_connection(conn)

async def create_yookassa_payment(
    user_id: int,
    credits: int,
    amount_rub: float,
    description: str,
    customer_email: str = "test@mail.ru",
):
    """Create YooKassa payment."""
    try:
        payment_uuid = str(uuid.uuid4())

        payment = Payment.create(
            {
                "amount": {"value": f"{amount_rub:.2f}", "currency": "RUB"},
                "confirmation": {
                    "type": "redirect",
                    "return_url": f"https://t.me/{BOT_USERNAME}",
                },
                "capture": True,
                "description": description,
                "metadata": {"user_id": str(user_id), "credits": str(credits)},
                "receipt": {
                    "customer": {
                        "email": customer_email
                    },
                    "items": [
                        {
                            "description": description,
                            "quantity": "1.00",
                            "amount": {"value": f"{amount_rub:.2f}", "currency": "RUB"},
                            "vat_code": "1",
                        }
                    ],
                },
            },
            payment_uuid,
        )

        return {
            "success": True,
            "payment_id": payment.id,
            "confirmation_url": payment.confirmation.confirmation_url,
        }
    except Exception as e:
        logger.error(f"Error creating YooKassa payment: {e}")
        return {"success": False, "error": str(e)}

async def poll_yookassa_payment(
    payment_id: str,
    user_id: int,
    chat_id: int,
    message_id: int,
    credits: int,
    timeout_minutes: int = 15,
):
    """Poll YooKassa payment status."""
    if payment_id in active_polls:
        logger.warning(f"Already polling YooKassa payment {payment_id}")
        return

    active_polls.add(payment_id)
    try:
        start_time = datetime.now()
        logger.info(f"Started polling YooKassa payment {payment_id} for user {user_id}, credits: {credits}")

        await ensure_user_exists(user_id, None)

        while (datetime.now() - start_time).total_seconds() < timeout_minutes * 60:
            try:
                payment = Payment.find_one(payment_id)

                if payment.status == "succeeded":
                    logger.info(f"YooKassa payment {payment_id} succeeded, processing")

                    await add_credits(user_id, credits, "yookassa", float(payment.amount.value))

                    success_text = (
                        f"🎉 Платеж успешно обработан\n"
                        f"Номер платежа: `{payment_id}`\n\n"
                        f"💳 Сумма: {payment.amount.value} рублей\n"
                        f"✅ Начислено: {credits} кредитов\n"
                    )

                    try:
                        if message_id:
                            try:
                                await bot.delete_message(chat_id=chat_id, message_id=message_id)
                            except Exception as e:
                                logger.warning(f"Could not delete payment message: {e}")

                        await bot.send_message(
                            chat_id=chat_id,
                            text=success_text,
                            parse_mode="Markdown",
                            message_effect_id="5046509860389126442",
                        )
                    except Exception as e:
                        logger.error(f"Error sending success message to user {user_id}: {e}")
                        return

                    return

                elif payment.status == "canceled":
                    logger.info(f"YooKassa payment {payment_id} was canceled")
                    break

            except Exception as e:
                logger.error(f"Error polling YooKassa payment {payment_id}: {e}")

            await asyncio.sleep(5)

        logger.info(f"YooKassa payment {payment_id} polling timeout")
        failure_text = "❌ Время ожидания оплаты истекло"
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="Попробовать снова", callback_data=f"retry_yookassa_{credits}")]
            ]
        )

        try:
            if message_id:
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=failure_text,
                    reply_markup=keyboard,
                )
            else:
                await bot.send_message(chat_id=user_id, text=failure_text, reply_markup=keyboard)
        except Exception as e:
            logger.error(f"Error sending failure message to user {user_id}: {e}")

    except Exception as e:
        logger.error(f"Error in YooKassa polling: {e}")
    finally:
        active_polls.discard(payment_id)

async def upload_prompt_image_to_r2(file_path: str, object_name: str) -> Optional[str]:
    """Upload prompt image to R2"""
    try:
        loop = asyncio.get_running_loop()
        
        def _upload():
            with open(file_path, 'rb') as f:
                r2_client.upload_fileobj(
                    f, R2_BUCKET_NAME, object_name,
                    ExtraArgs={'ContentType': 'image/jpeg'}
                )
            return f"{R2_PUBLIC_URL}/{object_name}"
        
        url = await loop.run_in_executor(None, _upload)
        logger.info(f"Uploaded prompt image to R2: {object_name}")
        return url
    except Exception as e:
        logger.error(f"Error uploading to R2: {e}")
        return None

async def upload_image_to_r2(file_path: str) -> Optional[str]:
    """Upload user image to R2 and return public URL"""
    try:
        timestamp = datetime.now().timestamp()
        object_name = f"user_photos/{uuid.uuid4()}_{timestamp}.jpg"
        url = await upload_prompt_image_to_r2(file_path, object_name)
        
        if url:
            logger.info(f"Uploaded user image to R2: {url}")
            return url
        else:
            logger.error("Failed to upload user image to R2")
            return None
    except Exception as e:
        logger.error(f"Error uploading user image to R2: {e}")
        return None

async def create_kia_task(image_url: str, prompt: str) -> Dict[str, Any]:
    """Create a KIA.AI nano-banana-edit generation task"""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {KIA_API_KEY}",
        }

        payload = {
            "model": KIA_MODEL,
            "input": {
                "prompt": prompt,
                "image_urls": [image_url],
                "output_format": "jpeg",
                "image_size": "1:1"
            },
        }

        logger.info(f"Creating KIA.AI task with prompt: {prompt[:100]}")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{KIA_API_URL}/createTask",
                headers=headers,
                json=payload
            ) as response:
                response_text = await response.text()

                if response.status == 200:
                    data = await response.json()
                    if data.get("code") == 200:
                        task_id = data["data"]["taskId"]
                        logger.info(f"KIA.AI task created successfully: {task_id}")
                        return {
                            "success": True,
                            "task_id": task_id,
                            "error_code": None,
                            "error_msg": None,
                        }
                    elif data.get("code") == 402:
                        logger.error("KIA.AI credits depleted!")
                        return {
                            "success": False,
                            "task_id": None,
                            "error_code": 402,
                            "error_msg": "Insufficient credits",
                        }
                    else:
                        logger.error(f"KIA.AI API error: {data}")
                        return {
                            "success": False,
                            "task_id": None,
                            "error_code": data.get("code"),
                            "error_msg": data.get("msg"),
                        }
                else:
                    logger.error(f"HTTP error from KIA.AI: {response.status}, {response_text}")
                    return {
                        "success": False,
                        "task_id": None,
                        "error_code": response.status,
                        "error_msg": response_text,
                    }
    except Exception as e:
        logger.error(f"Exception in create_kia_task: {e}", exc_info=True)
        return {
            "success": False,
            "task_id": None,
            "error_code": None,
            "error_msg": str(e),
        }

async def check_kia_task_status(task_id: str) -> Optional[Dict]:
    """Check the status of a KIA.AI generation task"""
    try:
        headers = {"Authorization": f"Bearer {KIA_API_KEY}"}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{KIA_API_URL}/recordInfo",
                headers=headers,
                params={"taskId": task_id}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    response_text = await response.text()
                    logger.error(f"HTTP error checking KIA.AI task: {response.status}, {response_text}")
                    return None
    except Exception as e:
        logger.error(f"Error checking KIA.AI task status: {e}")
        return None

async def poll_kia_task(task_id: str, max_wait: int = 300) -> Optional[str]:
    """Poll KIA.AI task until completion or timeout"""
    start_time = datetime.now()
    poll_interval = 5  # Start with 5 seconds
    
    while (datetime.now() - start_time).total_seconds() < max_wait:
        status_data = await check_kia_task_status(task_id)
        
        if not status_data or status_data.get("code") != 200:
            logger.error(f"Failed to check task status: {status_data}")
            return None
        
        task_data = status_data.get("data", {})
        state = task_data.get("state")
        
        if state == "success":
            result_json = task_data.get("resultJson")
            if result_json:
                try:
                    result = json.loads(result_json)
                    result_urls = result.get("resultUrls", [])
                    if result_urls:
                        logger.info(f"Task {task_id} completed successfully")
                        return result_urls[0]
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse resultJson: {e}")
                    return None
        
        elif state == "fail":
            fail_msg = task_data.get("failMsg", "Unknown error")
            logger.error(f"Task {task_id} failed: {fail_msg}")
            return None
        
        await asyncio.sleep(poll_interval)
        poll_interval = min(poll_interval + 1, 10)
    
    logger.error(f"Task {task_id} timed out after {max_wait} seconds")
    return None

async def download_image(url: str, file_path: str) -> bool:
    """Download image from URL"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    with open(file_path, "wb") as f:
                        f.write(await response.read())
                    return True
                else:
                    logger.error(f"HTTP error downloading image: {response.status}")
                    return False
    except Exception as e:
        logger.error(f"Error downloading image: {e}")
        return False

async def get_celebrities() -> list:
    """Get active celebrities from prompts table"""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT name FROM prompts 
                WHERE type = 'celebrity' AND is_active = TRUE 
                ORDER BY name
            """)
            results = cur.fetchall()
            return [row['name'] for row in results] if results else []
    except Exception as e:
        logger.error(f"Error fetching celebrities: {e}")
        return []
    finally:
        return_db_connection(conn)

async def get_actions() -> list:
    """Get active actions from prompts table"""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT name FROM prompts 
                WHERE type = 'action' AND is_active = TRUE 
                ORDER BY name
            """)
            results = cur.fetchall()
            return [row['name'] for row in results] if results else []
    except Exception as e:
        logger.error(f"Error fetching actions: {e}")
        return []
    finally:
        return_db_connection(conn)

def create_celebrity_keyboard(page: int = 0) -> InlineKeyboardMarkup:
    """Create paginated celebrity selection keyboard"""
    return InlineKeyboardMarkup(inline_keyboard=[[]])

async def create_celebrity_keyboard_async(page: int = 0) -> InlineKeyboardMarkup:
    """Create paginated celebrity selection keyboard from database"""
    celebrities = await get_celebrities()
    
    if not celebrities:
        return InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="❌ Нет доступных знаменитостей", callback_data="no_celebrities")
        ]])
    
    per_page = 10
    start = page * per_page
    end = start + per_page
    
    buttons = []
    celebs = celebrities[start:end]
    for i in range(0, len(celebs), 2):
        row = [InlineKeyboardButton(text=celebs[i], callback_data=f"celeb_{celebs[i]}")]
        if i + 1 < len(celebs):
            row.append(InlineKeyboardButton(text=celebs[i+1], callback_data=f"celeb_{celebs[i+1]}"))
        buttons.append(row)
    
    buttons.append([InlineKeyboardButton(text="🎲 Случайный выбор", callback_data="celeb_random")])
    
    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton(text="◀️", callback_data=f"celeb_page_{page-1}"))
    if end < len(celebrities):
        nav_buttons.append(InlineKeyboardButton(text="▶️", callback_data=f"celeb_page_{page+1}"))
    
    if nav_buttons:
        buttons.append(nav_buttons)
    
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def create_action_keyboard(celebrity: str) -> InlineKeyboardMarkup:
    """Create action selection keyboard"""
    return InlineKeyboardMarkup(inline_keyboard=[[]])

async def create_action_keyboard_async(celebrity: str) -> InlineKeyboardMarkup:
    """Create action selection keyboard from database"""
    actions = await get_actions()
    
    if not actions:
        return InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="❌ Нет доступных действий", callback_data="no_actions")
        ]])
    
    buttons = []
    actions_list = actions[:10]
    for i in range(0, len(actions_list), 2):
        row = [InlineKeyboardButton(text=actions_list[i], callback_data=f"action_{actions_list[i]}")]
        if i + 1 < len(actions_list):
            row.append(InlineKeyboardButton(text=actions_list[i+1], callback_data=f"action_{actions_list[i+1]}"))
        buttons.append(row)
    
    buttons.append([
        InlineKeyboardButton(text="🎲 Случайное", callback_data="action_random"),
        InlineKeyboardButton(text="◀️ Назад", callback_data="back_to_celebrity")
    ])
    
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def create_payment_keyboard(user_id: int) -> InlineKeyboardMarkup:
    """Create payment options keyboard"""
    buttons = [
        [
            InlineKeyboardButton(text=f"💳 10 - {PACKAGE_10['price_rub']}₽", callback_data="pay_10_rub"),
            InlineKeyboardButton(text=f"⭐ {PACKAGE_10['price_stars']}", callback_data="pay_10_stars")
        ],
        [
            InlineKeyboardButton(text=f"💳 50 - {PACKAGE_50['price_rub']}₽", callback_data="pay_50_rub"),
            InlineKeyboardButton(text=f"⭐ {PACKAGE_50['price_stars']}", callback_data="pay_50_stars")
        ],
        [
            InlineKeyboardButton(text=f"💳 100 - {PACKAGE_100['price_rub']}₽", callback_data="pay_100_rub"),
            InlineKeyboardButton(text=f"⭐ {PACKAGE_100['price_stars']}", callback_data="pay_100_stars")
        ],
        [InlineKeyboardButton(text=f"🎁 +{REFERRAL_BONUS} за друга", callback_data="invite_friend")]
    ]
    
    return InlineKeyboardMarkup(inline_keyboard=buttons)

@dp.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    """Handle /start command"""
    user_id = message.from_user.id
    username = message.from_user.username or message.from_user.first_name
    
    args = message.text.split()
    referred_by = args[1] if len(args) > 1 else None
    
    await ensure_user_exists(user_id, username, referred_by)
    
    try:
        collage_path = "examples/collage.jpg"
        if os.path.exists(collage_path):
            await message.answer_photo(
                FSInputFile(collage_path),
                caption=(
                    f"Привет, {message.from_user.first_name}! 🤳\n\n"
                    f"Отправь фото, где ты смотришь в камеру, чтобы сделать снимок со знаменитостью!"
                )
            )
        else:
            await message.answer(
                f"Привет, {message.from_user.first_name}! 🤳\n\n"
                f"Отправь фото, где ты смотришь в камеру, чтобы сделать снимок со знаменитостью!"
            )
    except Exception as e:
        logger.error(f"Error sending collage: {e}")
        await message.answer(
            f"Привет, {message.from_user.first_name}! 🤳\n\n"
            f"Отправь фото, где ты смотришь в камеру, чтобы сделать снимок со знаменитостью!"
        )
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📖 Инструкция", callback_data="instruction")]
    ])
    
    await message.answer(
        "💡 Нажми кнопку ниже для инструкции",
        reply_markup=keyboard
    )

@dp.callback_query(F.data == "instruction")
async def show_instruction(callback: CallbackQuery):
    """Show instruction"""
    user_id = callback.from_user.id
    
    await callback.answer(
        "1. Пожалуйста, убедитесь, что фото чёткое и не размытое, а в кадре нет посторонних людей\n"
        "2. Лицо должно быть хорошо видно целиком — важно, чтобы его ничего не закрывало\n"
        "⚠️ Обратите внимание: нейросеть с вами не знакома, поэтому при некорректно отправленной"
        "фотографии она может неточно передать особенности вашего лица",
        show_alert=True
    )

@dp.message(F.photo)
async def handle_photo(message: Message, state: FSMContext):
    """Handle user photo"""
    user_id = message.from_user.id
    
    photo = message.photo[-1]
    await state.update_data(photo_file_id=photo.file_id)
    await state.set_state(GenerationStates.choosing_celebrity)
    
    keyboard = await create_celebrity_keyboard_async(0)
    await message.answer(
        "Отлично! Теперь выбери знаменитость:",
        reply_markup=keyboard
    )

@dp.callback_query(F.data.startswith("celeb_page_"))
async def celebrity_page(callback: CallbackQuery):
    """Handle celebrity pagination"""
    user_id = callback.from_user.id
    page = int(callback.data.split("_")[-1])
    keyboard = await create_celebrity_keyboard_async(page)
    await callback.message.edit_reply_markup(reply_markup=keyboard)
    await callback.answer()

@dp.callback_query(F.data.startswith("celeb_"))
async def choose_celebrity(callback: CallbackQuery, state: FSMContext):
    """Handle celebrity selection"""
    user_id = callback.from_user.id
    
    if callback.data == "celeb_random":
        celebrities = await get_celebrities()
        if not celebrities:
            await callback.answer("❌ Нет доступных знаменитостей", show_alert=True)
            return
        celebrity = random.choice(celebrities)
    else:
        celebrity = callback.data.replace("celeb_", "")
    
    await state.update_data(celebrity=celebrity)
    await state.set_state(GenerationStates.choosing_action)
    
    keyboard = await create_action_keyboard_async(celebrity)
    await callback.message.edit_text(
        f"Выбрана знаменитость: {celebrity}\n\nТеперь выбери действие:",
        reply_markup=keyboard
    )
    await callback.answer()

@dp.callback_query(F.data == "back_to_celebrity")
async def back_to_celebrity(callback: CallbackQuery, state: FSMContext):
    """Go back to celebrity selection"""
    user_id = callback.from_user.id
    
    await state.set_state(GenerationStates.choosing_celebrity)
    keyboard = await create_celebrity_keyboard_async(0)
    await callback.message.edit_text(
        "Выбери знаменитость:",
        reply_markup=keyboard
    )
    await callback.answer()

@dp.callback_query(F.data.startswith("action_"))
async def choose_action(callback: CallbackQuery, state: FSMContext):
    """Handle action selection and generate image"""
    user_id = callback.from_user.id
    
    if callback.data == "action_random":
        actions = await get_actions()
        if not actions:
            await callback.answer("❌ Нет доступных действий", show_alert=True)
            return
        action = random.choice(actions)
    else:
        action = callback.data.replace("action_", "")
    
    data = await state.get_data()
    celebrity = data.get('celebrity')
    photo_file_id = data.get('photo_file_id')
    
    user_data = await get_user_data(user_id)
    if not user_data:
        await callback.answer("Ошибка: пользователь не найден", show_alert=True)
        return
    
    total_credits = user_data['credits'] + user_data['additional_credits']
    
    if total_credits < 1:
        await callback.message.edit_text(
            "❌ У вас недостаточно кредитов для генерации.\n\n"
            "Выберите пакет для покупки:",
            reply_markup=create_payment_keyboard(user_id)
        )
        await callback.answer()
        return
    
    if not await deduct_credits(user_id, 1):
        await callback.answer("Ошибка при списании кредитов", show_alert=True)
        return
    
    await callback.message.edit_text("⏳ Генерирую изображение, подождите...")
    await callback.answer()
    
    try:
        file = await bot.get_file(photo_file_id)
        file_path = f"/tmp/{user_id}_{datetime.now().timestamp()}.jpg"
        await bot.download_file(file.file_path, file_path)
        
        progress_msg = await callback.message.edit_text("⏳ Загружаю фото...")
        image_url = await upload_image_to_r2(file_path)
        
        if not image_url:
            await add_credits(user_id, 1, "refund", 0)
            await progress_msg.edit_text("❌ Ошибка при загрузке фото. Кредиты возвращены.")
            await state.clear()
            return
        
        prompt = f"A photo of a person with {celebrity}, {action}"
        
        await progress_msg.edit_text("⏳ Создаю задачу генерации...")
        task_result = await create_kia_task(image_url, prompt)
        
        if not task_result["success"]:
            await add_credits(user_id, 1, "refund", 0)
            error_msg = task_result.get("error_msg", "Unknown error")
            await progress_msg.edit_text(f"❌ Ошибка API: {error_msg}\nКредиты возвращены.")
            await state.clear()
            return
        
        task_id = task_result["task_id"]
        
        await progress_msg.edit_text("⏳ Генерирую изображение (это может занять до 3 минут)...")
        result_url = await poll_kia_task(task_id, max_wait=300)
        
        if not result_url:
            await add_credits(user_id, 1, "refund", 0)
            await progress_msg.edit_text("❌ Не удалось сгенерировать изображение. Кредиты возвращены.")
            await state.clear()
            return
        
        await progress_msg.edit_text("⏳ Загружаю результат...")
        result_path = f"/tmp/result_{user_id}_{datetime.now().timestamp()}.jpg"
        
        if not await download_image(result_url, result_path):
            await add_credits(user_id, 1, "refund", 0)
            await progress_msg.edit_text("❌ Ошибка при загрузке результата. Кредиты возвращены.")
            await state.clear()
            return
        
        result_text = (
            f"✅ Готово!\n\n"
            f"Знаменитость: {celebrity}\n"
            f"Действие: {action}\n\n"
            f"Осталось кредитов: {total_credits - 1}"
        )
        
        continue_keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Создать еще", callback_data="create_more")],
            [InlineKeyboardButton(text="💳 Купить кредиты", callback_data="buy_credits")],
            [InlineKeyboardButton(text="🎁 Пригласить друга", callback_data="invite_friend")]
        ])
        
        await progress_msg.delete()
        
        await callback.message.answer_photo(
            FSInputFile(result_path),
            caption=result_text,
            reply_markup=continue_keyboard
        )
        
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(result_path):
            os.remove(result_path)
        
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        await add_credits(user_id, 1, "refund", 0)
        await callback.message.answer("❌ Ошибка при генерации. Кредиты возвращены.")
    
    await state.clear()

@dp.callback_query(F.data == "create_more")
async def create_more(callback: CallbackQuery, state: FSMContext):
    """Start new generation"""
    user_id = callback.from_user.id
    
    await callback.message.answer("Отправь новое фото для генерации!")
    await callback.answer()

@dp.callback_query(F.data == "buy_credits")
async def buy_credits_callback(callback: CallbackQuery):
    """Show payment options"""
    user_id = callback.from_user.id
    
    await callback.message.answer(
        "💳 Выберите пакет кредитов:",
        reply_markup=create_payment_keyboard(user_id)
    )
    await callback.answer()

@dp.callback_query(F.data == "invite_friend")
async def invite_friend(callback: CallbackQuery):
    """Show referral link"""
    user_id = callback.from_user.id
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT referral_code, referrals FROM users WHERE user_id = %s", (user_id,))
            result = cur.fetchone()
            if result:
                ref_code = result['referral_code']
                ref_count = result['referrals']
                ref_link = f"https://t.me/{BOT_USERNAME}?start={ref_code}"
                
                text = (
                    f"🎁 <b>Пригласи друзей и получи бонусы!</b>\n\n"
                    f"За каждого приглашенного друга ты получишь <b>+{REFERRAL_BONUS} генерации</b>\n\n"
                    f"Твоя реферальная ссылка:\n"
                    f"<code>{ref_link}</code>\n\n"
                    f"Приглашено друзей: {ref_count}"
                )
                
                keyboard = InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(
                        text="📤 Поделиться ссылкой",
                        url=f"https://t.me/share/url?url={ref_link}&text=Попробуй создать фото со знаменитостью!"
                    )]
                ])
                
                await callback.message.answer(text, parse_mode="HTML", reply_markup=keyboard)
    finally:
        return_db_connection(conn)
    
    await callback.answer()

@dp.callback_query(F.data.startswith("pay_"))
async def handle_payment(callback: CallbackQuery):
    """Handle payment selection"""
    user_id = callback.from_user.id
    data = callback.data.split("_")
    amount = int(data[1])
    payment_type = data[2]  # 'rub' or 'stars'
    
    if amount == 10:
        package = PACKAGE_10
    elif amount == 50:
        package = PACKAGE_50
    else:
        package = PACKAGE_100
    
    if payment_type == "stars":
        prices = [LabeledPrice(label=f"{package['credits']} генераций", amount=package['price_stars'])]
        
        await bot.send_invoice(
            chat_id=user_id,
            title=f"{package['credits']} генераций",
            description=f"Покупка {package['credits']} генераций",
            payload=f"credits_{package['credits']}",
            provider_token="",  # Empty for Stars
            currency="XTR",
            prices=prices
        )
    else:
        if not yookassa_enabled:
            prepared_text = f"Здравствуйте, не могу оплатить через ЮKassa, хочу купить {package['credits']} генераций за {package['price_rub']} рублей"
            encoded_text = urllib.parse.quote(prepared_text)
            telegram_url = f"https://t.me/share/url?text={encoded_text}"
            keyboard = InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text="Связаться с поддержкой", url=telegram_url)]
                ]
            )
            await callback.message.answer(
                "💳 Оплата через ЮKassa временно недоступна. Свяжитесь с поддержкой.",
                reply_markup=keyboard
            )
            await callback.answer()
            return
        
        result = await create_yookassa_payment(
            user_id,
            package['credits'],
            float(package['price_rub']),
            f"{package['credits']} генераций"
        )
        
        if result["success"]:
            text = "Нажмите кнопку «Оплатить» ниже, затем вернитесь в бот — зачисление придёт автоматически."
            
            keyboard = InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text="Перейти к оплате", url=result["confirmation_url"])],
                    [InlineKeyboardButton(text="◀️ Назад", callback_data="buy_credits")]
                ]
            )
            
            sent_msg = await callback.message.answer(text, reply_markup=keyboard)
            
            asyncio.create_task(
                poll_yookassa_payment(
                    result["payment_id"],
                    user_id,
                    callback.message.chat.id,
                    sent_msg.message_id,
                    package['credits']
                )
            )
        else:
            await callback.message.answer(f"❌ Ошибка создания платежа: {result.get('error', 'Unknown')}")
    
    await callback.answer()

@dp.callback_query(F.data.startswith("retry_yookassa_"))
async def handle_retry_yookassa(callback: CallbackQuery):
    """Handle retry YooKassa payment - fixed"""
    credits = int(callback.data.split("_")[2])
    
    new_data = f"pay_{credits}_rub"
    callback.data = new_data
    
    await handle_payment(callback)

@dp.pre_checkout_query()
async def pre_checkout_handler(pre_checkout_query: PreCheckoutQuery):
    """Handle pre-checkout query"""
    await bot.answer_pre_checkout_query(pre_checkout_query.id, ok=True)

@dp.message(F.successful_payment)
async def successful_payment(message: Message):
    """Handle successful payment"""
    user_id = message.from_user.id
    payment = message.successful_payment
    
    credits = int(payment.invoice_payload.split("_")[1])
    
    if credits == 10:
        package = PACKAGE_10
    elif credits == 50:
        package = PACKAGE_50
    else:
        package = PACKAGE_100
    
    await add_credits(user_id, credits, "stars", package['price_stars'])
    
    await message.answer(
        f"✅ Оплата прошла успешно!\n\n"
        f"Начислено: {credits} генераций\n",
        # f"Теперь можешь создавать фото со знаменитостями 🎉",
        message_effect_id="5046509860389126442"
    )

async def main():
    """Main bot function"""
    dp.message.middleware(RateLimitMiddleware())
    dp.callback_query.middleware(RateLimitMiddleware())
    dp.callback_query.middleware(CallbackSecurityMiddleware())
    
    logger.info("Bot started")
    
    try:
        await dp.start_polling(bot, allowed_updates=["message", "callback_query", "pre_checkout_query"])
    finally:
        await bot.session.close()
        db_pool.closeall()

if __name__ == "__main__":
    asyncio.run(main())
