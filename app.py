import config
import predict
import os
from pathlib import Path
import numpy as np
from flask import Flask, request, abort
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageMessage, ImageSendMessage
)


SAVE_DIR = "./static/images"
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

app = Flask(__name__, static_folder="static", static_url_path="")

line_bot_api = LineBotApi(config.LINE_CHANNEL_ACCESS_TOKEN)  # config.pyで設定したチャネルアクセストークン
handler = WebhookHandler(config.LINE_CHANNEL_SECRET)  # config.pyで設定したチャネルシークレット

message_temp = "name: {}\nscore: {}%"
SRC_IMG_PATH = "static/images/{}.jpg"


@app.route("/")
def index():
    return "Hello World!"


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    message_id = event.message.id
    src_img_path = SRC_IMG_PATH.format(message_id)

    save_img(message_id, src_img_path)
    reply = calc_score(src_img_path)

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )

    Path(SRC_IMG_PATH.format(message_id)).absolute().unlink()
    print(message_id)


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    if event.reply_token == "00000000000000000000000000000000":
        return

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=event.message.text)
    )


# 画像の保存
def save_img(message_id, src_img_path):
    # message_idから画像のバイナリデータを取得
    message_content = line_bot_api.get_message_content(message_id)
    with open(src_img_path, "wb") as f:
        # バイナリを1024バイトずつ書き込む
        for chunk in message_content.iter_content():
            f.write(chunk)


# 予測スコアを算出
def calc_score(src_img_path):
    pred_name, pred_score = predict.pred_actress(src_img_path)
    pred_score = str(np.round(pred_score.numpy() * 100, 1))[1:-1]

    reply = message_temp.format(pred_name, pred_score)
    return reply


if __name__ == "__main__":
    app.debug = True   # デバッグモード有効化
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))