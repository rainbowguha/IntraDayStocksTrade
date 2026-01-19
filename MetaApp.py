import numpy as np
import os
import pytz
import time
import datetime
from Params import Strategy_On_params , GetHistory
from StrategyRep import PredictorEngine
from datetime import datetime
import requests
import smtplib
from email.mime.text import MIMEText
from Params import Weight

class MetaApi:
    Symbol_historyUpdates = []

    def __init__(self):
        self.time_zone = pytz.timezone('Asia/kolkata')
        self.unique_ticker = None
        self.error = None
        self.retry_count = 5
        self.models = {}
        self.Signals = {}
        self.sl_points ={}
        self.symbol_list = np.unique([ticker for ticker in Strategy_On_params])
        self.load_Strategy()
        self.Refresh_Var()

        #   sending email response :
        self.bot_name='NSE-GPT(AI DRIVEN MODEL)'
        self.sender_mail='algomate399@gmail.com'
        self.sender_pass='eddx tpyl sggx ryfr'
        self.recipient_mail='tapasguha258@gmail.com'

        msg='Engine refreshed @ :{}'.format(datetime.now(self.time_zone))
        # self.send_email_notification(msg)

    def Refresh_Var(self) :
        self.Symbol_historyUpdates=[]
        self.error=None
        self.Signals={}
        self.sl_points = {}

        # removing redundant file from the database_fx
        for symbol in self.symbol_list:
            file_path=os.path.join('database_fx' , symbol , '{}.csv'.format(symbol))
            if os.path.exists(file_path) :
                os.remove(file_path)

    def load_Strategy(self):

        try:
           for ticker in self.symbol_list:
               Strategy_ = Strategy_On_params[ticker]
               for key in Strategy_:
                   name = key['strategy'] + '_' + ticker
                   self.models[name] = PredictorEngine(name , ticker , key['Components'] , key['interval'])

        except Exception as e :
            self.error='Error:@load_strategy:{}'.format(e)
            print(self.error)

    def UpdateHistory(self):

        for symbol in self.symbol_list :
            u=0
            while u < self.retry_count :
                try :
                    file_path=os.path.join('database_fx' , symbol , '{}.csv'.format(symbol))
                    history=GetHistory(symbol)
                    history = history[history.index.date!=datetime.now(self.time_zone).today().date()]
                    history.to_csv(file_path)
                    self.Symbol_historyUpdates.append(symbol)
                    time.sleep(2)
                    break
                except Exception as e :
                    u+=1
                    time.sleep(5)
                    if u == self.retry_count :
                        self.error='Error:@UpdateHistory:{}:{}'.format(symbol , e)
                        print(self.error)

    def GenerateSignals(self):
        try:
            self.Signals = {}
            Updated_symbol = list(set(self.symbol_list) & set(self.Symbol_historyUpdates))

            for ticker in Updated_symbol:
                SIG=0
                SL={}
                for key , model in self.models.items():
                    if key.split('_')[-1] ==ticker:
                        sig , sl=model.GetPrediction()
                        SIG+=sig
                        SL[sig]=sl
                        time.sleep(3)

                if SIG:
                    self.Signals[ticker]=SIG
                    self.sl_points[ticker]=SL[1 if SIG > 0 else -1] if SIG else 0

            # print(self.Signals)
        except Exception as e:
            self.error="Error:@GEN_SIGNALS:{}".format(e)
            print(self.error)

    def place_order(self):

        InitialCapital=100000
        margin_x=5
        AvailableMargin=InitialCapital * margin_x

        ID_ = {'AXIS' : 'AXIS' ,'TCS' : 'TCS' ,'Adaniports':'ADANI' ,'MM':'MM' ,'Reliance':'REL' , 'CIPLA':'CIPLA' , 'JSWSTEEL':'JSW'}

        for symbol , signal in self.Signals.items():

            token = 'b80dee13-9db5-4bc5-b96a-845c4da8ed76'

            if not signal :
                continue

            trade_data={
                f"SL_{ID_[symbol]}" : self.sl_points[symbol] ,
                f"V_{ID_[symbol]}" : Weight[symbol] * AvailableMargin ,
                f"Y_PRED_{ID_[symbol]}" : 1 if signal > 0 else -1 ,
            }

            for key , val in trade_data.items() :
                url=f"https://api.tradetron.tech/api?auth-token={token}&key={key}&value={val}"
                # print(url)
                requests.get(url)

            time.sleep(3)

    def send_email_notification(self ,subject):

        msg_body = (f"Bot:{self.bot_name}\n"
                    f"Subject:{subject}")

        msg = MIMEText(msg_body)
        msg['Subject'] = f'{self.bot_name}:Response as on:'
        msg['From'] = self.sender_mail
        msg['To'] = self.recipient_mail

        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(self.sender_mail,self.sender_pass)
                server.sendmail(self.sender_mail, self.recipient_mail, msg.as_string())

        except Exception as e:
            self.error = f'Error:{e}'
            print(self.error)

