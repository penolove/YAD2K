import os
import json

from linebot import LineBotApi
from linebot.models import ImageSendMessage

class LineImageSender(object):
    def __init__(self, line_broadcast_path):
        """
        init a LineImageSender, used to sent image out

        Parameters
        ----------
        line_broadcast_path: str
            path of line_broadcast setting
            require key:
            - USER_IDS: audience
            - CHANNEL_ACCESS_TOKEN: line channel access token
            - SITE_DOMAIN: https domain url of static files
        """
        line_json_path = os.path.expanduser(line_broadcast_path)
        with open(line_json_path) as f:
            line_settings = json.load(f)
        self.line_audience_user_ids = line_settings['USER_IDS']
        channel_access_token = line_settings['CHANNEL_ACCESS_TOKEN']
        self.line_bot_api = LineBotApi(channel_access_token)
        self.site_domain = line_settings['SITE_DOMAIN']
        self.classes_to_sent = set(line_settings.get('CLASSES_TO_SENT', ['person']))

    def object_check_and_sent(self, detected_classes, img_path):
        """
        check if detected classes overlap with self.classes_to_sent
        if overlapped -> sent

        Parameters
        ----------
        detected_classes: set[str]
            detected_classes
        img_path: str
            image to sent
        """
        if self.classes_to_sent.intersection(detected_classes):
            self.send_img(img_path)

    def send_img(self, img_path):
        image_url = self.site_domain + img_path
        image_msg = ImageSendMessage(original_content_url = image_url,
                                     preview_image_url = image_url)
        self.line_bot_api.multicast(self.line_audience_user_ids, image_msg)

