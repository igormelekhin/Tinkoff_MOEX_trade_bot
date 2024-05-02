import tinkoff.invest
import decimal
import pickle
import os
import time
import datetime, pytz
import uuid

import APIUserInfo
import dataStorage
import featuresCreation
import fitPredict

class TradeBot(): #class to trade stocks using predicts from trained models
  def __init__(self, start=True, botDir="./botDir"):
    if not os.path.exists(botDir):
      os.mkdir(botDir)
    self.botDir = botDir
    #load models and params
    self.loadModels()
    #load token and account ids
    self.token = APIUserInfo.getToken()
    self.app_name = APIUserInfo.getAppname()
    if self.token == "": return;
    self.accountIDs = APIUserInfo.getAccountIDs()
    if self.accountIDs == []: return
    self.data = dataStorage.DataStorage()
    self.feats = featuresCreation.FeaturesCreation(self.data, self.params)
    self.feats.getInstInfo(os.path.join(self.botDir, "feats.pickle"), rewrite=False)
    self.predicting = fitPredict.StocksFitPredict(self.feats, self.params)
    #download trading info for the current day
    self.metainfoUpdated = None
    with tinkoff.invest.Client(self.token, target=tinkoff.invest.constants.INVEST_GRPC_API_SANDBOX, app_name=self.app_name) as self.grpcClient:
      while True:
        if self.updateTradingMetainfo() == 0: #no error
          break
        time.sleep(10)
    self.loadPredicts()
    if start:
      self.runBot()

  def runBot(self):
    #main loop
    self.bNeedToStop = False
    self.mainLoop()

  ############################################################################################################

  def mainLoop(self):
    #Run events
    while True:
      with tinkoff.invest.Client(self.token, target=tinkoff.invest.constants.INVEST_GRPC_API_SANDBOX, app_name=self.app_name) as self.grpcClient:
        wait = self.timeEvent()
      if wait <= 0 or self.bNeedToStop or os.path.exists("./stopBot.flag"):
        print("MAIN LOOP STOPPED")
        break
      if wait < 100: print("Waiting for", wait, "seconds")
      else: print("Waiting for", wait // 60, "minutes")
      time.sleep(wait)

	############################################################################################################

  def updatePositionsAndOrders(self):
    try:
      self.positions = []
      self.orders = []
      self.stopOrders = []
      for accountID in self.accountIDs:
        response = self.grpcClient.operations.get_positions(account_id = accountID)
        self.positions.append(response)
        response = self.grpcClient.orders.get_orders(account_id = accountID)
        self.orders.append(response.orders)
      return 0
    except Exception as e:
      print("!!! updatePositionsAndOrders error:", str(e))
      return -1

	############################################################################################################

  def updateTradingMetainfo(self, force=False):
    if (self.metainfoUpdated is None) or (self.metainfoUpdated.date() < tinkoff.invest.utils.datetime.now().date()):
      try:
        self.exchangeEndTime = {}
        response = self.grpcClient.instruments.trading_schedules()
        for schedule in response.exchanges:
          if schedule.exchange == "MOEX":
            self.MOEX_start = schedule.days[0].start_time
          self.exchangeEndTime[schedule.exchange] = schedule.days[0].end_time
        self.instMetainfo = self.data.getInstrumentsMetainfo(force=True)
        self.metainfoUpdated = tinkoff.invest.utils.datetime.utcnow().replace(tzinfo=pytz.timezone('UTC'))
        return 0
      except Exception as e:
        print("!!! updateTradingMetainfo error:", str(e))
        return -1
    else:
      return 0

	############################################################################################################

  def loadModels(self):
    #Load models for predicts
    try:
      print("Loading trained models..")
      self.models = []
      modelFilename = os.path.join(self.botDir, "models.pickle")
      with open(modelFilename, "rb") as f:
        trainedInfo = pickle.load(f)
      for modelNum in [1, 5]:
        self.models.append((trainedInfo["models"][modelNum][0], trainedInfo["models"][modelNum][1]))
      self.params = trainedInfo["params"]
      print("Done loading trained models")
      return 0
    except Exception as e:
      print("!!! loadModels error:", str(e), "!!!")
      return -1

	############################################################################################################
  
  def loadPredicts(self):
    PREDICTS_FILENAME = os.path.join(self.botDir, "predicts.pickle")
    if os.path.exists(PREDICTS_FILENAME):
      with open(PREDICTS_FILENAME, "rb") as f:
        self.predicts = pickle.load(f)
    else:
      self.predicts = []

  ############################################################################################################

  def calcPredictsForToday(self):
    if (len(self.predicts) > 0) and (self.predicts[-1][0] > self.MOEX_start):
      return
    #calculate predicts
    print("Calculating predicts...")
    dateToXPredFrames, lastDate = self.feats.createXPredDataset(lastDays=1)
    predictsForToday = []
    for accountNum in range(len(self.accountIDs)):
      models = self.models[accountNum][0]
      predictsForToday.append(self.predicting.runPredict(dateToXPredFrames, [lastDate], self.models[accountNum][0], self.models[accountNum][1]))
    self.predicts.append((tinkoff.invest.utils.datetime.utcnow().replace(tzinfo=pytz.timezone('UTC')), predictsForToday))
    PREDICTS_FILENAME = os.path.join(self.botDir, "predicts.pickle")
    with open(PREDICTS_FILENAME, "wb") as f:
      pickle.dump(self.predicts, f)
    print("Done calculating predicts")

	############################################################################################################

  def groupInfoForTodayActions(self):
    #Pack current positions and orders according to predicts
    self.instActions = [{} for ID in self.accountIDs]
    if not((len(self.predicts) > 0) and (self.predicts[-1][0] > self.MOEX_start)):
      return
    for accountNum in range(len(self.accountIDs)):
      for predict in self.predicts[-1][1][accountNum]:
        exchange = self.data.instMetainfo[predict["figi"]].exchange
        if not exchange in self.exchangeEndTime: exchangeEndTime = self.exchangeEndTime["MOEX"]
        else: exchangeEndTime = self.exchangeEndTime[exchange]
        if exchangeEndTime.timestamp() == 0: continue
        self.instActions[accountNum][predict["figi"]] = {}
        self.instActions[accountNum][predict["figi"]]["predict"] = predict
        self.instActions[accountNum][predict["figi"]]["endTime"] = exchangeEndTime
        self.instActions[accountNum][predict["figi"]]["position"] = 0
        self.instActions[accountNum][predict["figi"]]["buyOrder"] = None
        self.instActions[accountNum][predict["figi"]]["sellOrder"] = None
      for position in self.positions[accountNum].securities:
        self.instActions[accountNum].setdefault(position.figi, {"buyOrder" : None, "sellOrder" : None})
        self.instActions[accountNum][position.figi]["position"] = position.balance + position.blocked
      for order in self.orders[accountNum]:
        if order.lots_executed >= order.lots_requested: continue
        if order.execution_report_status == tinkoff.invest.OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_REJECTED: continue
        if order.execution_report_status == tinkoff.invest.OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_CANCELLED: continue
        self.instActions[accountNum].setdefault(order.figi, {"position" : 0, "buyOrder" : None, "sellOrder" : None})
        if order.direction == tinkoff.invest.OrderDirection.ORDER_DIRECTION_BUY:
          self.instActions[accountNum][order.figi]["buyOrder"] = order
        if order.direction == tinkoff.invest.OrderDirection.ORDER_DIRECTION_SELL:
          self.instActions[accountNum][order.figi]["sellOrder"] = order

	############################################################################################################

  def timeEvent(self):
    timeNow = tinkoff.invest.utils.datetime.utcnow().replace(tzinfo=pytz.timezone('UTC'))
    print("=======")
    print(timeNow.time(), "utc time")
    if (self.updateTradingMetainfo() != 0): #error in getting schedules and stocks metainfo for today
      return 10;
    if self.MOEX_start.timestamp() == 0 or self.MOEX_start.weekday() == 5:
      print("No trading today")
      return 60 * 60 * 5
    if self.updatePositionsAndOrders() != 0: #error in information update
      return 10;
    self.groupInfoForTodayActions()
    ########## PREPARATION STAGE <10:00
    if timeNow < self.MOEX_start:
      print("PREPARATION STAGE")
      pass
      return max(int((self.MOEX_start - timeNow).total_seconds()) // 2 ,1)
    ########## BUY STAGE 10:00 - 11:00
    if timeNow >= self.MOEX_start and timeNow < self.MOEX_start + datetime.timedelta(minutes=60):
      print("BUY STAGE")
      #if no canldes for today download it
      if self.data.needNewCandles():
        print("Downloading candles...")
        if (self.data.downloadCandles(verb=0)) != 0: #error
          return 15
        print("Done downloading candles")
        #process new candles into features
        self.feats.getInstInfo(os.path.join(self.botDir, "feats.pickle"))
      #if no predicts - do it and put into file
      if not ((len(self.predicts) > 0) and (self.predicts[-1][0] > self.MOEX_start)):
        self.calcPredictsForToday()
      #group predict, orders and position information for each used instrument
      self.groupInfoForTodayActions()

      for accountNum in range(len(self.accountIDs)):
        #check for not sold positions
        bSoldExtraPositions = False
        for inst in self.instActions[accountNum]:
          if (not "predict" in self.instActions[accountNum][inst]): #position from previous days should be sold
            sellLots = self.instActions[accountNum][inst]["position"] // self.data.instMetainfo[inst].lot
            if sellLots > 0:
              orderID = str(uuid.uuid4())
              print("Making sell order", orderID)
              response = self.grpcClient.orders.post_order(
                      quantity = sellLots,
                      direction = tinkoff.invest.OrderDirection.ORDER_DIRECTION_SELL,
                      account_id = self.accountIDs[accountNum],
                      order_type = tinkoff.invest.OrderType.ORDER_TYPE_MARKET,
                      order_id = orderID,
                      instrument_id = inst,
                  )
              print(response)
              bSoldExtraPositions = True
        if bSoldExtraPositions: #wait for sell and reload positions
          return 1
        #get current bank to be spent
        bank = 0
        for money in self.positions[accountNum].money:
          if money.currency == "rub":
            bank = money.units + money.nano * 10**(-9)
        #make required buy and sell orders
        for inst in self.instActions[accountNum]:
          if (self.instActions[accountNum][inst]["buyOrder"] is None) and (self.instActions[accountNum][inst]["position"] == 0):
            #get bank part and appropriate lots amount
            buyLots = int((bank * self.instActions[accountNum][inst]["predict"]["bankPart"] / self.instActions[accountNum][inst]["predict"]["buyPrice"]) / self.data.instMetainfo[inst].lot)
            if buyLots > 0:
              orderID = str(uuid.uuid4())
              print("Making buy order", orderID, buyLots, tinkoff.invest.utils.decimal_to_quotation(decimal.Decimal(self.instActions[accountNum][inst]["predict"]["buyPrice"])), inst)
              response = self.grpcClient.orders.post_order(
                        quantity = buyLots,
                        price = tinkoff.invest.utils.decimal_to_quotation(decimal.Decimal(self.instActions[accountNum][inst]["predict"]["buyPrice"])),
                        direction = tinkoff.invest.OrderDirection.ORDER_DIRECTION_BUY,
                        account_id = self.accountIDs[accountNum],
                        order_type = tinkoff.invest.OrderType.ORDER_TYPE_LIMIT,
                        order_id = orderID,
                        instrument_id = inst,
                  )
              print(response)
          #buy order fullfilled
          if (self.instActions[accountNum][inst]["sellOrder"] is None) and (self.instActions[accountNum][inst]["buyOrder"] is None) and (self.instActions[accountNum][inst]["position"] > 0):
            sellLots = self.instActions[accountNum][inst]["position"] // self.data.instMetainfo[inst].lot
            if sellLots > 0:
              orderID = str(uuid.uuid4())
              print("Making sell order", orderID)
              response = self.grpcClient.orders.post_order(
                        quantity = sellLots,
                        price = tinkoff.invest.utils.decimal_to_quotation(decimal.Decimal(self.instActions[accountNum][inst]["predict"]["sellPrice"])),
                        direction = tinkoff.invest.OrderDirection.ORDER_DIRECTION_SELL,
                        account_id = self.accountIDs[accountNum],
                        order_type = tinkoff.invest.OrderType.ORDER_TYPE_LIMIT,
                        order_id = orderID,
                        instrument_id = inst,
                  )
              print(response)

      return 60 * 5

    ########## SELL STAGE 11:00 - 18:40/23:50
    if timeNow >= self.MOEX_start + datetime.timedelta(minutes=60):
      print("SELL STAGE")
      for accountNum in range(len(self.accountIDs)):
        bCancelledStaledOrders = False
        for order in self.orders[accountNum]:
          if order.direction == tinkoff.invest.OrderDirection.ORDER_DIRECTION_BUY:
            print("Cancelling staled buy order " + order.order_id + " for", order.figi, "for", accountNum, self.accountIDs[accountNum])
            response = self.grpcClient.orders.cancel_order(
                  account_id = self.accountIDs[accountNum],
                  order_id = order.order_id,
              )
            print(response)
            bCancelledStaledOrders = True
        if bCancelledStaledOrders: #wait, update info and run again
          return 1

        #exchange closing time
        bCancelledStaledOrders = False
        for inst in self.instActions[accountNum]:
          if timeNow > (self.instActions[accountNum][inst]["endTime"] - tinkoff.invest.utils.timedelta(minutes=15)):
            if (self.instActions[accountNum][inst]["position"] > 0) and (self.instActions[accountNum][inst]["sellOrder"] is not None):
              print("Cancelling staled sell order ")
              response = self.grpcClient.orders.cancel_order(
                    account_id = self.accountIDs[accountNum],
                    order_id = self.instActions[accountNum][inst]["sellOrder"].order_id,
                )
              print(response)
              bCancelledStaledOrders = True

        if bCancelledStaledOrders: #wait, update info and run again
          return 1

        bSoldStaledPositions = False
        for inst in self.instActions[accountNum]:
          if timeNow > (self.instActions[accountNum][inst]["endTime"] - tinkoff.invest.utils.timedelta(minutes=15)):
            if self.instActions[accountNum][inst]["position"] > 0:
              print("Market price sell on exchange closing for", inst)
              sellLots = self.instActions[accountNum][inst]["position"] // self.data.instMetainfo[inst].lot
              if sellLots > 0:
                orderID = str(uuid.uuid4())
                print("Making market price sell order", orderID)
                response = self.grpcClient.orders.post_order(
                      quantity = sellLots,
                      direction = tinkoff.invest.OrderDirection.ORDER_DIRECTION_SELL,
                      account_id = self.accountIDs[accountNum],
                      order_type = tinkoff.invest.OrderType.ORDER_TYPE_MARKET,
                      order_id = orderID,
                      instrument_id = inst,
                  )
              print(response)
              bSoldStaledPositions = True

          if bSoldStaledPositions: #wait, update info and run again
            return 1

      bAnyPositions = False
      for inst in self.instActions[accountNum]:
        for accountNum in range(len(self.accountIDs)):
          if self.instActions[accountNum][inst]["position"] > 0:
            bAnyPositions = True
      if not bAnyPositions:
        print("No positions to sell anymore")
        return 60 * 60 * 2

    return 60 * 10

	############################################################################################################

  def sellAllPositions(self):
    if self.updateTradingMetainfo() != 0: #error
      return -1
    with tinkoff.invest.Client(self.token, target=tinkoff.invest.constants.INVEST_GRPC_API_SANDBOX, app_name = self.app_name) as client:
      for accountNum in range(len(self.accountIDs)):
        for position in self.positions[accountNum].securities:
          sellLots = position.balance // self.data.instMetainfo[position.figi].lot
          if sellLots > 0:
            orderID = str(uuid.uuid4())
            print("Selling", position.figi, "for market price")
            response = client.orders.post_order(
                  quantity = sellLots,
                  direction = tinkoff.invest.OrderDirection.ORDER_DIRECTION_SELL,
                  account_id = self.accountIDs[accountNum],
                  order_type = tinkoff.invest.OrderType.ORDER_TYPE_MARKET,
                  order_id = orderID,
                  instrument_id = position.figi,
              )
            print(response)
    return 0

	############################################################################################################

  def cancelAllOrders(self):
    if self.updateTradingMetainfo() != 0: #error
      return -1
    with tinkoff.invest.Client(self.token, target=tinkoff.invest.constants.INVEST_GRPC_API_SANDBOX, app_name = self.app_name) as client:
      for accountNum in range(len(self.accountIDs)):
        for order in self.orders[accountNum]:
          print("Cancelling order " + order.order_id + " for", order.figi, "for", accountNum, self.accountIDs[accountNum])
          response = client.orders.cancel_order(
                account_id = self.accountIDs[accountNum],
                order_id = order.order_id,
            )
          print(response)
    return 0

if __name__ == "__main__":
	pass
	
	
