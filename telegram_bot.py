"""Telegram bot: on-demand Skew-T soundings via Simple_Sounding.py.

Usage (in a chat with the bot):
    /sounding                      -> latest NKX sounding
    /sounding 72572                -> latest sounding for station 72572
    /sounding NKX 2025-01-01T12    -> that station's sounding at that run

Setup:
    export TELEGRAM_BOT_TOKEN="<token from @BotFather>"
    python telegram_bot.py

No chat ID is needed -- this replies wherever the command came from,
it doesn't push anywhere on its own.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

from telegram import BotCommand, Update
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
SOUNDING_SCRIPT = SCRIPT_DIR / 'Simple_Sounding.py'


async def sounding(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args
    station = args[0] if len(args) >= 1 else 'NKX'
    cmd = [sys.executable, str(SOUNDING_SCRIPT), '--station', station]
    if len(args) >= 2:
        cmd += ['--datetime', args[1]]

    await update.message.reply_text(f'Fetching {station} sounding...')

    try:
        result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True,
                                text=True, timeout=120)
    except subprocess.TimeoutExpired:
        await update.message.reply_text('Timed out fetching/plotting that sounding.')
        return

    if result.returncode != 0:
        error_line = result.stderr.strip().splitlines()[-1] if result.stderr else 'unknown error'
        logger.warning('Simple_Sounding.py failed: %s', result.stderr)
        await update.message.reply_text(f'Failed: {error_line}')
        return

    # Simple_Sounding.py names its output <station>_<run>.png in SCRIPT_DIR;
    # the most recently modified matching file is the one just made.
    png_files = sorted(SCRIPT_DIR.glob(f'{station}_*.png'), key=lambda f: f.stat().st_mtime)
    if not png_files:
        await update.message.reply_text('Ran, but no image was produced.')
        return

    with open(png_files[-1], 'rb') as photo:
        await update.message.reply_photo(photo)


async def post_init(application: Application) -> None:
    # Registers the "/" autocomplete menu Telegram clients show while
    # typing -- without this the command still works, but no hint pops
    # up. Uses the default scope, which covers both DMs and group chats
    # (the bot still needs to actually be a member of the group, and its
    # commands there may need "@YourBotUsername" if another bot in the
    # same group also defines a /sounding command).
    await application.bot.set_my_commands([
        BotCommand('sounding', 'Skew-T sounding: /sounding [station] [datetime]'),
    ])


def main() -> None:
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    if not token:
        raise SystemExit('Set TELEGRAM_BOT_TOKEN in the environment first.')

    app = Application.builder().token(token).post_init(post_init).build()
    app.add_handler(CommandHandler('sounding', sounding))
    logger.info('Bot starting (polling)...')
    app.run_polling()


if __name__ == '__main__':
    main()
