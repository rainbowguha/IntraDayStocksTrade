from PredictorEngine import *
from Params import *
import time
from datetime import datetime
import requests
from fyersApi import HIST_BROKER_

class MetaApi:
    def __init__(self):
        self.mods = {}
        self.error = None
        self.retry_count = 5
        self.symbol_list = np.unique([v for v in strategies.values()])
        self._ON_SYMBOL_ = []
        self.__SIGNALS__ = {}
        self.weight_params = {}
        self.Weights = {}
        self.sl_params = {}
        self.load_str()

    def load_str(self):
         try:
             for s , symbols in strategies.items():
                 self.mods = {f'{s}_{symbol}' : PredictorEngine(s , symbol) for symbol in symbols}

         except Exception as e:
             self.error='Error:@load_strategy:{}'.format(e)
             print(self.error)

    def UpdateHistory(self) :
        market = HIST_BROKER_().login()

        for symbol in self.symbol_list :
            dir_path=os.path.join('database_fx' , symbol)
            os.makedirs(dir_path , exist_ok=True)
            file_path=os.path.join(dir_path , '{}'.format(symbol))

            for attempt in range(1 , self.retry_count+1) :
                try :
                    history=GetHistory(market , symbol)
                    if history is None or history.empty :
                        raise ValueError('GetHistory returned no data')

                    today=datetime.now(time_zone).date()
                    history=history[history.index.date != today]
                    history.to_csv(file_path)
                    self._ON_SYMBOL_.append(symbol)
                    time.sleep(2)
                    break

                except Exception as e :
                    if attempt == self.retry_count :
                        self.error='Error:@UpdateHistory:{}:{}'.format(symbol , e)
                        print(self.error)
                    else :
                        time.sleep(5)

    def GenerateSignals(self):

        self.__SIGNALS__ ,  self.sl_params = {} , {}

        if not self._ON_SYMBOL_:
            return self.__SIGNALS__

        for __STR__ , mod in self.mods.items():
            symbol = __STR__.split('_')[1]
            if symbol in self._ON_SYMBOL_:
                self.__SIGNALS__[__STR__]  , self.sl_params[__STR__] = mod.__GET_SIG__()

    def place_order(self) :

        for __STR__ , signal in self.__SIGNALS__.items() :
            symbol = __STR__.split('_')[1]

            if not signal :
                continue

            trade_data={
                f"MUL_{ID[symbol]}" : self.sl_params[__STR__]['multiplier'] ,
                f"LK_{ID[symbol]}" : self.sl_params[__STR__]['lookback'] ,
                f"Y_PRED_{ID[symbol]}" : signal ,
            }

            for key , val in trade_data.items() :
                url=f"https://api.tradetron.tech/api?auth-token={token}&key={key}&value={val}"
                requests.get(url)

            time.sleep(3)
