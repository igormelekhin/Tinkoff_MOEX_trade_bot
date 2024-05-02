import tinkoff.invest
import pickle, os, datetime, time, pytz

import APIUserInfo

class DataStorage(): #class to download and store list of instruments and candles
	def __init__(self):
		self.token = APIUserInfo.getToken()
		self.getInstrumentsMetainfo()
		self.getSpreads(False)
		CANDLES_FILENAME = "data/candles.pickle"
		if os.path.exists(CANDLES_FILENAME):
			with open(CANDLES_FILENAME, "rb") as f:
				self.candles = pickle.load(f)
		else:
			self.candles = {}
			
	############################################################################################################		

	def getInstrumentsMetainfo(self, force=False):
		FILENAME = "data/instruments.pickle"
		if (not force) and os.path.exists(FILENAME):
			with open(FILENAME, "rb") as f:
				self.instMetainfo = pickle.load(f)
			return self.instMetainfo
		#download
		instMetainfoList = []
		with tinkoff.invest.Client(self.token, target=tinkoff.invest.constants.INVEST_GRPC_API_SANDBOX) as client:
			shares = client.instruments.shares()
			for share in shares.instruments:
				if share.currency == "rub" and share.real_exchange == tinkoff.invest.RealExchange.REAL_EXCHANGE_MOEX and share.buy_available_flag and share.sell_available_flag:
					instMetainfoList.append(share)
		#save
		self.instMetainfo = {}
		for inst in instMetainfoList:
			self.instMetainfo[inst.figi] = inst
		with open(FILENAME, "wb") as f:
			pickle.dump(self.instMetainfo, f)
		return self.instMetainfo
		
	############################################################################################################
		
	def getSpreads(self, update=True):
		FILENAME = "data/spreads.pickle"
		if os.path.exists(FILENAME):
			with open(FILENAME, "rb") as f:
				self.spreads = pickle.load(f)
		else:
			self.spreads = {}
		#download
		if update:
			now = tinkoff.invest.utils.now()
			with tinkoff.invest.Client(self.token, target=tinkoff.invest.constants.INVEST_GRPC_API_SANDBOX) as client:
				for inst in self.instMetainfo:
					res = client.market_data.get_order_book(instrument_id=inst, depth=1)
					if len(res.bids) == 0 or len(res.asks) == 0:
						continue
					self.spreads.setdefault(inst, [])
					self.spreads[inst].append((now, res.bids[0], res.asks[0]))
			#save
			with open(FILENAME, "wb") as f:
				pickle.dump(self.spreads, f)
		return self.spreads
		
	############################################################################################################

	def needNewCandles(self):
		DATE_FILENAME = "data/lastCandleDate.pickle"
		if os.path.exists(DATE_FILENAME):
			with open(DATE_FILENAME, "rb") as f:
				lastDate = pickle.load(f)
			if lastDate.date() == datetime.datetime.utcnow().replace(tzinfo=pytz.timezone('UTC')).date():
				return False
		return True

	############################################################################################################
		
	def downloadCandles(self, period = 150, retries=3, clear=False, verb=0):
		DATE_FILENAME = "data/lastCandleDate.pickle"
		if os.path.exists(DATE_FILENAME) and (not clear):
			with open(DATE_FILENAME, "rb") as f:
				lastDate = pickle.load(f)
			lastDate -= datetime.timedelta(days=2)
		else:
			lastDate = datetime.datetime(2012, 1, 1, 0, 0, 0, tzinfo=pytz.timezone('UTC'))
		upToDate = min(datetime.datetime.utcnow().replace(tzinfo=pytz.timezone('UTC')), lastDate + datetime.timedelta(days=2 + period))
		if verb > 0: print("dates", lastDate, upToDate)
		bFailed = False
		with tinkoff.invest.Client(self.token, target=tinkoff.invest.constants.INVEST_GRPC_API_SANDBOX) as client:
			for inst in self.instMetainfo:
				for retry in range(retries):
					if verb > 0: print(self.instMetainfo[inst].ticker, inst)
					try:
						candles = client.get_all_candles(instrument_id = inst,
														from_ = lastDate,
														to = upToDate,
														interval = tinkoff.invest.CandleInterval.CANDLE_INTERVAL_DAY,
														candle_source_type = tinkoff.invest.schemas.CandleSource.CANDLE_SOURCE_UNSPECIFIED)
						self.candles.setdefault(inst, [])
						candles = list(candles)
						if len(candles) == 0: break

						newDate = candles[0].time
						cutOldCandles = 0
						while (cutOldCandles < len(self.candles[inst])) and (self.candles[inst][-1-cutOldCandles].time >= newDate): cutOldCandles += 1
						if cutOldCandles > 0: self.candles[inst] = self.candles[inst][:-cutOldCandles]

						self.candles[inst].extend(candles)
						break
					except Exception as e: #exception during candles download
						errorMessage = str(e)
						if ("EXHAUSTED" in errorMessage): #just need to wait and try again
							SLEEP = 10
							time.sleep(SLEEP)
							print(errorMessage, "sleep for " + str(SLEEP) + " seconds")
							retry -= 1
							continue
            #another exception
						print("!!! downloadCandles error: " + str(e) + " for " + self.instMetainfo[inst].ticker + " " + inst)
						if retry < retries - 1:
							print("retrying")
						else:
							bFailed = True							
				if bFailed: #if failed all retries, dont try for other instruments
					break

		if not bFailed: #if everything is ok, save into files
			#update date
			with open(DATE_FILENAME, "wb") as f:
				pickle.dump(upToDate, f)
			CANDLES_FILENAME = "data/candles.pickle"
			with open(CANDLES_FILENAME, "wb") as f:
				pickle.dump(self.candles, f)
			print("done downloading till ", upToDate)
			return 0
		else:
			return -1
			
	############################################################################################################

	def removeDublicates(self):
		for inst in self.candles:
			newCandles = []
			datesUsed = set()
			for candle in reversed(self.candles[inst]):
				if candle.time in datesUsed:
					print("Dublicate removed")
					continue
				datesUsed.add(candle.time)
				newCandles.append(candle)
			self.candles[inst] = list(reversed(newCandles))
		CANDLES_FILENAME = "data/candles.pickle"
		with open(CANDLES_FILENAME, "wb") as f:
			pickle.dump(self.candles, f)

	############################################################################################################
		
if __name__ == "__main__":
	data = DataStorage()
	data.downloadCandles(verb=1, clear=False)
