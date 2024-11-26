import aiohttp
import json
import time
import hashlib
import hmac
import base64
from env import SMS_SERVICE_ID, SMS_ACCESS_KEY, SMS_PHONE_NUMBER, SMS_SECRET_KEY

async def sms_send(sendSmsList, content):
    url1 = "https://sens.apigw.ntruss.com"
    uri2 = "/sms/v2/services/" + SMS_SERVICE_ID + "/messages"
    api_url = url1 + uri2
    timestamp = str(int(time.time() * 1000))
    string_to_sign = "POST " + uri2 + "\n" + timestamp + "\n" + SMS_ACCESS_KEY
    signature = make_signature(string_to_sign)

    headers = {
        'Content-Type': "application/json; charset=UTF-8",
        'x-ncp-apigw-timestamp': timestamp,
        'x-ncp-iam-access-key': SMS_ACCESS_KEY,
        'x-ncp-apigw-signature-v2': signature
    }

    body = {
        "type": "SMS",
        "contentType": "COMM",
        "from": SMS_PHONE_NUMBER,
        "content": content,
        "countryCode": "82"
    }
    body['messages'] = []
    # 전화번호 목록에 대해 일괄전송
    for phoneNumber in sendSmsList:
        body['messages'].append(
            {
            "to": phoneNumber,
            "subject": "안전관리 위험 알림",
            "content": content
            }
        )
    body = json.dumps(body)
    async with aiohttp.ClientSession(headers=headers) as session:
        response = await session.post(api_url, data=body)
        response.raise_for_status()


def make_signature(string):
    secret_key = bytes(SMS_SECRET_KEY, 'UTF-8')
    string = bytes(string, 'UTF-8')
    string_hmac = hmac.new(secret_key, string, digestmod=hashlib.sha256).digest()
    string_base64 = base64.b64encode(string_hmac).decode('UTF-8')
    return string_base64