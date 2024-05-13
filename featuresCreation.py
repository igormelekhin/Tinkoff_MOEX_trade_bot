import tinkoff.invest
import pickle, os, datetime, time, pytz
import numpy as np

import defaultParameters

def getDerivativeWeights(d, sigma=0.1, count=0):
  w = [1]
  if count == 0:
    count = 1000
  for i in range(1, count):
    w.append(-w[-1] * (d-i+1) / (i))
    if np.abs(w[-1] / w[-2]) < sigma:
      break
  if w[-1] == 0:
    w = w[:-1]
  return w
  
ws = []
ws.append(getDerivativeWeights(0.1, 0.01, 30))
ws.append(getDerivativeWeights(0.15, 0.01, 30))
ws.append(getDerivativeWeights(0.25 0.01, 30))
ws.append(getDerivativeWeights(0.35, 0.01, 30))
ws.append(getDerivativeWeights(0.5, 0.01, 30))
ws.append(getDerivativeWeights(0.7, 0.01, 30))



class FeaturesCreation(): #class to process candles to make features for train, test and real time prediction
  def __init__(self, data, params):
    self.data = data
    self.candles = data.candles
    self.instMetainfo = data.instMetainfo
    self.currentParams = params.copy()
    for p in defaultParameters.defaultParams:
      if not p in self.currentParams:
        self.currentParams[p] = defaultParameters.defaultParams[p]
    
  ############################################################################################################
    
  def estimateSpreads(self, thr=0.75):
    self.estimatedSpread = {}
    for inst in self.data.spreads:
      values = []
      for measure in self.data.spreads[inst]:
        buyPrice = measure[1].price.units + measure[1].price.nano*10**(-9)
        sellPrice = measure[2].price.units + measure[2].price.nano*10**(-9)
        values.append((sellPrice - buyPrice) / (sellPrice+buyPrice))
      values = sorted(values)
      self.estimatedSpread[inst] = values[-1 - int( (1-thr)*len(values) )]
    return self.estimatedSpread
    
  ############################################################################################################
    
  def getInstInfo(self, storeFilename="", rewrite=False):
    if storeFilename != "" and os.path.exists(storeFilename) and rewrite == False:
      with open(storeFilename, "rb") as f:
        storedInfo = pickle.load(f)
    else:
      storedInfo = {"passedCandles" : {}, "instInfo" : {}}
        
    passedCandles = storedInfo["passedCandles"]
    instInfo = storedInfo["instInfo"]
        
    for inst in self.candles:
      if not inst in instInfo: #init
        if len(self.candles[inst]) < 60: continue
        instInfo[inst] = {}
        length = len(self.candles[inst]) + 365
        instInfo[inst]["candleTime"] = [0] * length
        instInfo[inst]["open"] = np.zeros(length)
        instInfo[inst]["high"] = np.zeros(length)
        instInfo[inst]["low"] = np.zeros(length)
        instInfo[inst]["close"] = np.zeros(length)
        instInfo[inst]["volume"] = np.zeros(length)
        instInfo[inst]["maxPrice"] = np.zeros(length)
        instInfo[inst]["maxVolume"] = np.zeros(length)
        instInfo[inst]["meanHigh"] = np.zeros(length)
        instInfo[inst]["openDeriv"] = np.zeros(length)
        instInfo[inst]["meanAbsRet"] = np.zeros(length)
        
      passedCandles.setdefault(inst, 2)
      for i in range(passedCandles[inst]-1, len(self.candles[inst])):
        passedCandles[inst] += 1
        candle = self.candles[inst][i]
        prevCandle = self.candles[inst][i-1]
        instInfo[inst]["candleTime"][i] = candle.time

        instInfo[inst]["open"][i] = candle.open.units + candle.open.nano * 10**(-9)
        instInfo[inst]["high"][i] = prevCandle.high.units + prevCandle.high.nano * 10**(-9)
        instInfo[inst]["low"][i] = prevCandle.low.units + prevCandle.low.nano * 10**(-9)
        instInfo[inst]["close"][i] = prevCandle.close.units + prevCandle.close.nano * 10**(-9)
        instInfo[inst]["volume"][i] = prevCandle.volume
        #max price
        instInfo[inst]["maxPrice"][i] = max(instInfo[inst]["maxPrice"][i-1], instInfo[inst]["open"][i])
        #max volume
        instInfo[inst]["maxVolume"][i] = max(instInfo[inst]["maxVolume"][i-1], instInfo[inst]["volume"][i])
        #open derivative
        if i >= 30:
          instInfo[inst]["openDeriv"][i] = (ws[2] * instInfo[inst]["open"][i-29:i+1]).sum()
          
        #mean abs return
        DAYS_RET = 200
        if i > DAYS_RET + 1:
          o1 = instInfo[inst]["open"][i-DAYS_RET-1]
          o2 = instInfo[inst]["open"][i-DAYS_RET]
          oldRet = (o2 / o1) - 1
          o1 = instInfo[inst]["open"][i-1]
          o2 = instInfo[inst]["open"][i]
          newRet = (o2 / o1) - 1
          instInfo[inst]["meanAbsRet"][i] = (instInfo[inst]["meanAbsRet"][i-1] * DAYS_RET + np.abs(newRet) - np.abs(oldRet)) / DAYS_RET
          if instInfo[inst]["meanAbsRet"][i] < 0: instInfo[inst]["meanAbsRet"][i] = 1 / 10**6
        elif i > 1:
          o1 = instInfo[inst]["open"][i-1]
          o2 = instInfo[inst]["open"][i]
          newRet = (o2 / o1) - 1
          instInfo[inst]["meanAbsRet"][i] = (instInfo[inst]["meanAbsRet"][i-1] * (i-1) + np.abs(newRet)) / i

        #mean high
        DAYS_HIGH = 120
        if i > DAYS_HIGH + 1:
          o = instInfo[inst]["open"][i-1]
          h = instInfo[inst]["high"][i]
          highChangeNow = (h / o) - 1
          
          if highChangeNow < 0:
            print(highChangeNow, "highChangeNow", o, h, inst, i)
            
          o = instInfo[inst]["open"][i-DAYS_HIGH-1]
          h = instInfo[inst]["high"][i-DAYS_HIGH]
          
          highChangeThen = (h / o) - 1
          
          if highChangeThen < 0:
            print(highChangeThen, "highChangeThen", o, h, inst, i)
          
          instInfo[inst]["meanHigh"][i] = (instInfo[inst]["meanHigh"][i-1] * DAYS_HIGH + highChangeNow - highChangeThen) / DAYS_HIGH
          if instInfo[inst]["meanHigh"][i] < 0: instInfo[inst]["meanHigh"][i] = 1 / 10**6
          
        elif i > 1:
          o = instInfo[inst]["open"][i-1]
          h = instInfo[inst]["high"][i]
          highChangeNow = (h / o) - 1
          if highChangeNow < 0:
            print(highChangeNow, "highChangeNow", o, h, inst, i)
          instInfo[inst]["meanHigh"][i] = (instInfo[inst]["meanHigh"][i-1] * (i-1) + highChangeNow) / i

      passedCandles[inst] -= 1

    #saving
    self.instInfo = instInfo
    self.passedCandles = passedCandles
    if storeFilename:
      with open(storeFilename, "wb") as f:
        pickle.dump(storedInfo, f)
        
  ############################################################################################################
  
  def retScale(self, realVal, instInfo, ind, params):
    return realVal
    P = 1 / instInfo["meanAbsRet"][ind]
    if realVal >= 0: integ = 0.5 + (1-np.exp(-P*realVal))*0.5
    else: integ = 0.5 - (1-np.exp(P*realVal))*0.5
    return integ
    
  ############################################################################################################
  
  def retUnscale(self, scaledVal, instInfo, ind, params):
    return scaledVal
    P = 1 / instInfo["meanAbsRet"][ind]
    if scaledVal >= 0.5: realVal = -np.log(1-(scaledVal-0.5)*2) * instInfo["meanAbsRet"][ind]
    else: realVal = np.log(1-(-scaledVal+0.5)*2) * instInfo["meanAbsRet"][ind]
    return realVal
    
  ############################################################################################################
  
  def highScale(self, realVal, instInfo, ind, params):
    P = 1 / instInfo["meanHigh"][ind]
    integ = 1 - np.exp(-P*realVal)
    return integ

  ############################################################################################################
  
  def highUnscale(self, scaledVal, instInfo, ind, params):
    P = 1 / instInfo["meanHigh"][ind]
    realVal = np.log(1-scaledVal) / (-P)
    return realVal
    
  ############################################################################################################
  
  def getBin(self, val, levels):
    for i in range(len(levels)):
      if val < levels[i]:
        return i
    return len(levels)
  
  ############################################################################################################

  def createXFrame(self, params, inst, ind):
    instInfo = self.instInfo[inst]
    #mean volumes for last 3 and 30 days
    lastVol = instInfo["volume"][ind-2:ind+1].mean()
    meanVol = instInfo["volume"][ind-29:ind+1].mean() + 1
    #open and volume ratios to max values
    if instInfo["open"][ind] == 0:
      print(instInfo["open"][ind], instInfo["maxPrice"][ind], inst, ind)
    toMax = instInfo["open"][ind] / instInfo["maxPrice"][ind]
    toMaxVol = instInfo["volume"][ind] / (instInfo["maxVolume"][ind] + 1)
    
    xFrame = [lastVol/meanVol, toMax, toMaxVol]
    
    xFrame.append(str(self.instMetainfo[inst].sector))
    xFrame.append(instInfo["candleTime"][ind].weekday())
    
    #scaled returns for different periods
    framesBack = [1, 2, 4, 10, 20]
    for j in framesBack:
      retVal = (instInfo["open"][ind] / instInfo["open"][ind-j]) - 1
      retVal = self.retScale(retVal, instInfo, ind, {})
      xFrame.append(retVal)
    #scaled returns for last few days
    for j in range(7):
      retVal = (instInfo["open"][ind-j] / instInfo["open"][ind-j-1]) - 1
      retVal = self.retScale(retVal, instInfo, ind, {})
      xFrame.append(retVal)
    #open price derivatives for last few days
    for j in range(7):
      xFrame.append(instInfo["openDeriv"][ind-j])
      
    #volume changes for different periods
    for j in framesBack:
      xFrame.append((instInfo["volume"][ind] - instInfo["volume"][ind-j]) / (instInfo["volume"][ind-j] + 1))
    #volume changes for last few days
    for j in range(7):
      xFrame.append((instInfo["volume"][ind-j] - instInfo["volume"][ind-j-1]) / (instInfo["volume"][ind-j-1] + 1))
    
    return xFrame
    
  ############################################################################################################
  
  def createYFrame(self, params, instInfo, ind, high):
    if high: #Y dataset for highs
      o = instInfo["open"][ind]
      h = instInfo["high"][ind+1]
      highScaled = self.highScale((h / o) - 1, instInfo, ind, params)
      yFrame = self.getBin(highScaled, params["highLevels"])
    else: #Y dataset for returns
      o1 = instInfo["open"][ind]
      o2 = instInfo["open"][ind+1]
      retScaled = self.retScale((o2 / o1) - 1, instInfo, ind, params)
      yFrame = self.getBin(retScaled, [-params["stopLoss"], *params["retLevels"]])  
    return yFrame
    
  ############################################################################################################

  def createXYTrainDatasets(self, params={}):
    if params == {}: params = self.currentParams
    print(params)
    dateToXFrames = {}
    dateToYRetFrames = {}
    dateToYHighFrames = {}
    for inst in self.instInfo:
      info = self.instInfo[inst]
      for i in range(30, self.passedCandles[inst]):
        date = info["candleTime"][i]
        xFrame = self.createXFrame(params, inst, i)
        dateToXFrames.setdefault(date, [])
        dateToXFrames[date].append({"x" : xFrame, "inst" : inst, "frameInd" : i})
        yFrame = self.createYFrame(params, info, i, high=False)
        dateToYRetFrames.setdefault(date, [])
        dateToYRetFrames[date].append({"y" : yFrame, "inst" : inst, "frameInd" : i})
        yFrame = self.createYFrame(params, info, i, high=True)
        dateToYHighFrames.setdefault(date, [])
        dateToYHighFrames[date].append({"y" : yFrame, "inst" : inst, "frameInd" : i})

    return dateToXFrames, dateToYRetFrames, dateToYHighFrames
    
  ############################################################################################################
  
  def createKFolds(self, dateToFrames, K=5, cutLastFrames=10, fromYear=2000, skipLastDays=0, params={}):
    minDate, maxDate = None, None
    for date in dateToFrames:
      if date.year < fromYear: continue
      if minDate is None: minDate, maxDate = date, date
      minDate = min(minDate, date)
      maxDate = max(maxDate, date)
    dayLength = (maxDate - minDate).days
    print(dayLength, "days length")
    days = []
    day = minDate
    maxDate = maxDate - datetime.timedelta(skipLastDays)
    while day < maxDate:
      days.append(day)
      day = day + datetime.timedelta(days = 1)
    chunks = []
    chunkSize = dayLength // K
    for i in range(K):
      chunks.append(days[chunkSize * i : chunkSize * (i+1)-cutLastFrames])

    folds = [] #train/test sets
    for test1 in range(K):
      for test2 in range(test1+1, K):
        folds.append([[], []])
        #train chunks
        for i in range(K):
          if i == test1 or i == test2: continue
          folds[-1][0].extend(chunks[i])
        #2 test chunks
        folds[-1][1] = chunks[test1].copy()
        folds[-1][1].extend(chunks[test2])
    
    self.kFolds = folds
    return folds

  ############################################################################################################

  def collectDataWithDatesList(self, dataByDates, datesList, column, needFullInfo=True, params={}):
    dataset = []
    fullInfo = []
    for date in datesList:
      if date in dataByDates:
        for frame in dataByDates[date]:
          dataset.append(frame[column])
          if needFullInfo: fullInfo.append(frame)
    return dataset, fullInfo

  ############################################################################################################

  def createXPredDataset(self, params={}, lastDays=120):
    dateToXFrames = {}
    minDate = datetime.datetime.utcnow().replace(tzinfo=pytz.timezone('UTC')) - datetime.timedelta(days=lastDays+1)
    lastDate = None
    for inst in self.instInfo:
      if self.passedCandles[inst] < lastDays + 30: continue
      info = self.instInfo[inst]
      for i in range(self.passedCandles[inst] - lastDays, self.passedCandles[inst]):
        date = info["candleTime"][i]
        if date < minDate: continue
        if lastDate is None: lastDate = date
        else: lastDate = max(lastDate, date)
        xFrame = self.createXFrame(params, inst, i)
        dateToXFrames.setdefault(date, [])
        dateToXFrames[date].append({"x" : xFrame, "inst" : inst, "frameInd" : i})  
    return dateToXFrames, lastDate

############################################################################################################

    
if __name__ == "__main__":
  pass
