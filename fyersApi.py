from datetime import timedelta

from fyers_apiv3 import fyersModel
import datetime
from time import sleep
import os
import pyotp
import requests
from urllib.parse import parse_qs,urlparse
import base64
import pytz
import pandas as pd


def getEncodedString(string) :
    string=str(string)
    base64_bytes=base64.b64encode(string.encode("ascii"))
    return base64_bytes.decode("ascii")


class HIST_BROKER_() :
    def __init__(self) :
        self.BROKER_APP=None
        self.BROKER_SOCKET=None
        self.access_token=None
        self.client_id=None
        self.time_zone=pytz.timezone('Asia/kolkata')

    def login(self) :

        redirect_uri="https://www.google.com/"
        self.client_id='GBF61JC6AF-100'
        secret_key='03HZE67VQN'
        FY_ID="XT00158"
        TOTP_KEY="5MI36QR765HXYCG2JMW5OE5SGPEQUBLC"
        PIN="2304"

        URL_SEND_LOGIN_OTP="https://api-t2.fyers.in/vagator/v2/send_login_otp_v2"
        res=requests.post(url=URL_SEND_LOGIN_OTP , json={"fy_id" : getEncodedString(FY_ID) , "app_id" : "2"}).json()
        if datetime.datetime.now(self.time_zone).second % 30 > 27 : sleep(5)
        URL_VERIFY_OTP="https://api-t2.fyers.in/vagator/v2/verify_otp"
        res2=requests.post(url=URL_VERIFY_OTP ,
                           json={"request_key" : res["request_key"] , "otp" : pyotp.TOTP(TOTP_KEY).now()}).json()

        ses=requests.Session()
        URL_VERIFY_OTP2="https://api-t2.fyers.in/vagator/v2/verify_pin_v2"
        payload2={"request_key" : res2["request_key"] , "identity_type" : "pin" , "identifier" : getEncodedString(PIN)}
        res3=ses.post(url=URL_VERIFY_OTP2 , json=payload2).json()
        ses.headers.update({
            'authorization' : f"Bearer {res3['data']['access_token']}"
        })

        TOKENURL="https://api-t1.fyers.in/api/v3/token"
        payload3={"fyers_id" : FY_ID ,
                  "app_id" : self.client_id[:-4] ,
                  "redirect_uri" : redirect_uri ,
                  "appType" : "100" , "code_challenge" : "" ,
                  "state" : "None" , "scope" : "" , "nonce" : "" , "response_type" : "code" , "create_cookie" : True}

        res3=ses.post(url=TOKENURL , json=payload3).json()
        url=res3['Url']
        parsed=urlparse(url)
        auth_code=parse_qs(parsed.query)['auth_code'][0]
        auth_code
        grant_type="authorization_code"
        response_type="code"
        session=fyersModel.SessionModel(
            client_id=self.client_id ,
            secret_key=secret_key ,
            redirect_uri=redirect_uri ,
            response_type=response_type ,
            grant_type=grant_type
        )

        session.set_token(auth_code)
        response=session.generate_token()
        self.access_token=response['access_token']
        self.BROKER_APP=fyersModel.FyersModel(client_id=self.client_id , is_async=False , token=self.access_token ,
                                              log_path=os.getcwd())
        return self.BROKER_APP


def fetch_ohlcv(broker_obj , ticker , limit=365):

    def historical_bydate(symbol , sd , ed , interval='D') :
        data={"symbol" : symbol , "resolution" : f'{interval}' , "date_format" : "1" , "range_from" : str(sd) ,
              "range_to" : str(ed) ,
              "cont_flag" : "1"}
        nx=broker_obj.history(data)
        cols=['datetime' , 'open' , 'high' , 'low' , 'close' , 'volume']
        df=pd.DataFrame.from_dict(nx['candles'])
        df.columns=cols
        df['datetime']=[pd.Timestamp.fromtimestamp(x) for x in df['datetime']]
        df=df.set_index('datetime')

        return df

    enddate = datetime.datetime.now().date()
    sd = enddate - timedelta(days=limit)
    df = pd.DataFrame()

    n = abs((sd - enddate).days)
    ab = None

    while ab == None:
        sd = (enddate - datetime.timedelta(days=n))
        ed = (sd + datetime.timedelta(days=99 if n > 100 else n)).strftime("%Y-%m-%d")
        sd = sd.strftime("%Y-%m-%d")
        dx = historical_bydate(ticker, sd, ed)
        df = pd.concat([df,dx],axis=0)
        n = n - 100 if n > 100 else n - n
        if n == 0:
            ab = "done"

    return df