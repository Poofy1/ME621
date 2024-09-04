# ME621

ML-powered Telegram bot for personalized e621.net image delivery.

ME621 is a comprehensive program designed to feed you hand picked e621.net content automatically. Using machine learning, it learns your tastes and delivers a personalized feed of images directly to your Telegram.

This is a WIP project, you will encounter unfinished features and issues.

## Features:

1. Label e621 images directly on the webui interface
2. Easily download user's favorited images on their e621 account
     - With the ability to remotely add/remove images on their e621 account
4. Automatically trains ML models and assists in the labeling process
5. Deploy a Telegram bot to send newly posted images that the model predicts the user will enjoy

## Requirements
- Recommended: NVIDIA GPU with at least 8GB VRAM
- e621.net account
- e621 API key
- Telegram account
- [Telegram bot API key](https://help.zoho.com/portal/en/kb/desk/support-channels/instant-messaging/telegram/articles/telegram-integration-with-zoho-desk#Telegram_Integration)

## Installation

1. Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/)
2. Clone the repository: `git clone https://github.com/Poofy1/ME621.git`
3. Run `webui.bat`
