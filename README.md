# ME621

Comprehensive program designed to feed you hand picked e621.net content automatically.

## Features:

1. Label e621 images directly on the webui interface
2. Easily download user's favorited images on their e621 account
     - With the ability to remotely add/remove images on their e621 account
4. Automatically trains ML models and assists in the labeling process
5. Deploy a Telegram bot to send newly posted images that the model predicts the user will enjoy

## Requirements
- Recommended: NVIDIA GPU with at least 8GB VRAM
- e621 account with API key
- Telegram account with Bot API key

## Installation

1. Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/)
2. Clone the repository: `git clone https://github.com/Poofy1/ME621.git`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run `webui.py`
