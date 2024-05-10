import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import random
import pickle
import tinkoff.invest
import os

import featuresCreation
import defaultParameters
 
from catboost import CatBoostClassifier, CatBoostRegressor, Pool, cv

def catBoostTrain(X, Y, X_val, Y_val, X_test, trainingParams, libSpecificParams, callbacks=[]):
    params = trainingParams.copy()
    defaultParams = {"ITERS" : 300, "LOSS" : "Poisson", "SEED" : 1, "CLASSIFIER" : False}
    for p in defaultParams:
      if not (p in params):
        params[p] = defaultParams[p]
    if "CLASSIFIER" in params and params["CLASSIFIER"]:
      cat = CatBoostClassifier(iterations=params["ITERS"], loss_function=params["LOSS"], use_best_model=True, thread_count=6, random_seed=params["SEED"], **libSpecificParams)
    else:
      cat = CatBoostRegressor(iterations=params["ITERS"], loss_function=params["LOSS"], use_best_model=True, thread_count=6, random_seed=params["SEED"], **libSpecificParams)
    sample_weight = np.ones(len(Y))
    if ("CLASS_WEIGHT" in trainingParams) and (trainingParams["CLASS_WEIGHT"] > 1):
        w = trainingParams["CLASS_WEIGHT"]
        for i in range(len(Y)):
            if Y[i] == 1:
                sample_weight[i] = w
    cat_feats = [] #handle str and other features
    for i in range(len(X[0])):
        if type(X[0][i]) == str:
            cat_feats.append(i)
    cat.fit(X, Y, cat_features=cat_feats, callbacks=callbacks, plot=False, verbose=False, eval_set=(X_val,Y_val), sample_weight=sample_weight)
    if params["CLASSIFIER"]:
      return cat.predict_proba(X_test), cat
    else: return cat.predict(X_test), cat


class StocksFitPredict(): #class to train models, perform trading simulation, estimate results, get predicts
  def __init__(self, featStorage, params = {}):
    self.featStorage = featStorage
    self.currentParams = params.copy()
    for p in defaultParameters.defaultParams:
      if not p in self.currentParams:
        self.currentParams[p] = defaultParameters.defaultParams[p]
    
  ############################################################################################################
  
  def calcGeomMeanOfOutcomes(self, outcomes):
    prod = 1
    summ = 0
    for o in outcomes:
      if o[0] < 0:
        print("o[0] < 0 !!!!")
        o = (0.00000001, o[1])
      if o[1] < 0:
        print("o[1] < 0 !!!!")
        o = (o[0], 0.00000001)
      if o[0] == 0:
        print("o[0] == 0")
      summ += o[1]
      prod *= o[0] ** o[1]
    return prod ** (1/summ)
  
  ############################################################################################################
  
  def calcOptimalTPandProfit(self, params, Y_preds_ret, Y_preds_high, Y_test_info, Y_pair_probs, spreads=None):
    TPs, expectedProfits = np.zeros(len(Y_preds_ret)), np.zeros(len(Y_preds_ret))
    highLevels, retLevels = params["highLevels"], [-params["stopLoss"], *params["retLevels"]]
  
    for i in range(len(Y_preds_ret)):
      bestTP = -1
      bestProfit = -1
      totalHighProb = Y_preds_high[i].sum()
      totalRetsProb = Y_preds_ret[i].sum()
      for TP_ind in range(len(highLevels)): #check TP levels
        outcomes = []
        for high_ind in range(len(highLevels)+1):
          for ret_ind in range(len(retLevels)+1):
            rawProb1 = Y_preds_ret[i][ret_ind] / totalRetsProb
            rawProb2 = Y_preds_high[i][high_ind] / totalHighProb
            pairProb = Y_pair_probs[i][high_ind][ret_ind] * rawProb2
            inst = Y_test_info[i]["inst"]
            instInfo = self.featStorage.instInfo[inst]
            frameInd = Y_test_info[i]["frameInd"]
            if TP_ind <= high_ind -1: #high bigger than TP and TP is taken
              outcomeRes = 1 + featuresCreation.FeaturesCreation.highUnscale(None, highLevels[TP_ind], instInfo, frameInd, params)
            else: #simple return
              if ret_ind == 0: #return so low, so its stop loss
                outcomeRes = 1 + featuresCreation.FeaturesCreation.retUnscale(None, retLevels[ret_ind], instInfo, frameInd, params)
              elif ret_ind == len(retLevels): #when its bigger than all levels
                outcomeRes = 1 + featuresCreation.FeaturesCreation.retUnscale(None, retLevels[ret_ind-1] * 1., instInfo, frameInd, params)
              else:
                outcomeRes = 1 + featuresCreation.FeaturesCreation.retUnscale(None, 0.5 * (retLevels[ret_ind-1] + retLevels[ret_ind]), instInfo, frameInd, params)
            outcomeProb = ((rawProb1 * rawProb2) * (pairProb ** params["pairProbPow"])) ** (1/(1+params["pairProbPow"]))
            if spreads is not None:
              outcomes.append([outcomeRes - spreads[Y_test_info[inst]], outcomeProb])
            else:
              outcomes.append([outcomeRes, outcomeProb])
        totalProb = 0
        for o in outcomes:
          totalProb += o[1]
        for j in range(len(outcomes)):
          outcomes[j] = (outcomes[j][0], outcomes[j][1] / totalProb)
        expectedProfit = self.calcGeomMeanOfOutcomes(outcomes) - 1
        
        if expectedProfit > bestProfit:
          bestProfit = expectedProfit
          bestTP = featuresCreation.FeaturesCreation.highUnscale(None, highLevels[TP_ind], instInfo, frameInd, params)
      TPs[i] = bestTP
      expectedProfits[i] = bestProfit
    return expectedProfits, TPs
  
  ############################################################################################################
  
  def calcBuysAndBankParts(self, params, dateFrames, Y_profit_preds, Y_TP_preds, ind, curBank, THR):
    buys = []
    for frame in dateFrames:
      if Y_profit_preds[ind] > THR:
        buyInfo = {"frameInfo" : frame, "frameInd" : ind, "TP" :  Y_TP_preds[ind], "rawBankPart" : (Y_profit_preds[ind]+0.1) ** params["partPower"]}
        buys.append((Y_profit_preds[ind], buyInfo))
      ind += 1

    #sort to get best buys. Take only required number of buys
    buys = sorted(buys, key=lambda buy: buy[0],  reverse=True)
    if len(buys) * params["BankPart"] > curBank:
      buys = buys[ : int(curBank / params["BankPart"]) + 1]
    if len(buys) > params["MaxBuys"]:
      buys = buys[:params["MaxBuys"]]
    
    total = 0
    for buy in buys:
      total += buy[1]["rawBankPart"]
    #disperse money
    for i in range(len(buys)):
      itsPart = buys[i][1]["rawBankPart"] / total
      buys[i][1]["bankPart"] = params["BankPart"] * len(buys) * itsPart
      
    return buys, ind
    
  ############################################################################################################
  
  def runBacktest(self, params, dates, dateToYRetFrames, Y_profit_preds, Y_TP_preds, refsList, instStats, logFile="", verb=0):
    stats = {"takeProfitsAmount" : 0, "stopLossesAmount" : 0, "profitDealsAmount" : 0, "lossDealsAmount" : 0}
    bank = 1
    expectedOutcomes, optimalTPs, predsByDays, buysOutcomes, dayOutcomes, history, refs = [], [], [], [], [], [], []
    
    newLogs2File = ""
    indBase, ind = 0, 0
    for date in dates:
      if date in dateToYRetFrames:
        bankAtDayStart = bank
        stDayOutcome = ""
        predsByDays.append([])
        ind2 = ind
        for frame in dateToYRetFrames[date]:
          predsByDays[-1].append(Y_profit_preds[ind2])
          ind2 += 1

        predsSoFar = []
        for dayFrames in predsByDays[max(0, len(predsByDays) - 1 - params["lastTopFrames"]) :]:
          predsSoFar.extend(dayFrames)
        predsSoFar = sorted(predsSoFar, reverse=False)
        if "topPreds" in params: THR = predsSoFar[int( len(predsSoFar) * params["topPreds"] )]
        else: THR = -10000
      
        indBase = ind
        buys, ind = self.calcBuysAndBankParts(params, dateToYRetFrames[date], Y_profit_preds, Y_TP_preds, ind, bank, THR)
      
        for buy in buys:
          buyInfo = buy[1]
          inst = buyInfo["frameInfo"]["inst"]
          instInfo = self.featStorage.instInfo[inst]
          frameInd = buyInfo["frameInfo"]["frameInd"]
          ticker = self.featStorage.instMetainfo[inst].ticker

          instStats.setdefault(ticker, {"boughtTimes" : 0, "totalResult" : 1})
          instStats[ticker]["boughtTimes"] += 1
          o, h, l, c = instInfo["open"][frameInd], instInfo["high"][frameInd+1], instInfo["low"][frameInd+1], instInfo["close"][frameInd+1]
          TAKE_PROFIT = max(params["minTakeProfit"], buyInfo["TP"])
          frameInd = buyInfo["frameInd"]
        
          expectedOutcomes.append(buy[0])
          optimalTPs.append(TAKE_PROFIT)

          TAKE_PROFIT *= params["takeProfitMul"]
          
          STOP_LOSS = params["stopLoss"]
              
          if (l / o) - 1 < -STOP_LOSS:
            outcome = 1 - STOP_LOSS
            stats["stopLossesAmount"] += 1
          elif (h / o) - 1 > TAKE_PROFIT:
            outcome = 1 + TAKE_PROFIT
            stats["takeProfitsAmount"] += 1
          else:
            outcome = (c / o)
            if c > o: stats["profitDealsAmount"] += 1
            else: stats["lossDealsAmount"] += 1
          
          buysOutcomes.append(outcome)
          instStats[ticker]["totalResult"] *= outcome
          stDayOutcome += ticker + " {:+.3f}%".format((outcome-1) * 100) + " ({:.2f})".format(buy[1]["bankPart"]) +"\n"

          bankTook = buy[1]["bankPart"] * bankAtDayStart
          bank -= bankTook

          bankTook *= 1 - params["comission"]
          bankGot = bankTook * outcome
          bankGot *= 1 - params["comission"]
          bank += bankGot
        
        if verb >= 2 or (verb > 0 and random.randint(0, 200) == 0):
          stDayOutcome = "\n" + str(date.date()) + " === " +  "{:+.3f}% ({:.3f})".format(100*((bank / bankAtDayStart) - 1), bank) + "\n" + stDayOutcome
          print(stDayOutcome)
        if logFile:
          stDayOutcome = "\n" + str(date.date()) + " === " +  "{:+.3f}% ({:.3f})".format(100*((bank / bankAtDayStart) - 1), bank) + "\n" + stDayOutcome
          newLogs2File += stDayOutcome + "\n\n"
        dayOutcomes.append((bank / bankAtDayStart) -1)
      else: #not date in dateToYFrames:
        if verb >= 2:
          stDayOutcome = "\n" + str(date.date()) + " === " +  "{:+.3f}% ({:.3f})".format(0, bank)
          print(stDayOutcome)
        if logFile:
          stDayOutcome = "\n" + str(date.date()) + " === " +  "{:+.3f}% ({:.3f})".format(0, bank)
          newLogs2File += stDayOutcome + "\n\n"
      history.append(bank)

    if verb > 0:
      print("Take Profit amount: \t", stats["takeProfitsAmount"])
      print("Stop Loss amount: \t", stats["stopLossesAmount"])
      print("Profit deals amount: \t", stats["profitDealsAmount"])
      print("Loss deals amount: \t", stats["lossDealsAmount"])
    return bank, history, refs, buysOutcomes, dayOutcomes, expectedOutcomes, optimalTPs, newLogs2File
        
  ############################################################################################################

  def estimateMetrics(self, Y_true, Y_preds):
    return sklearn.metrics.log_loss(Y_true, Y_preds)

  ############################################################################################################

  def runTrain(self, params={}, refs=[], verb=1, saveTo="", logFile="", debugTrain=False):
    if params == {}: params = self.currentParams
    currentParams = params
    if verb > 0:
      print(currentParams)

    #lists witch statistics for each fold and training with backtests in general
    expectedReturns, optimalTPs, expectedOutcomes, buysOutcomes, dayOutcomes = [], [], [], [], []
    metricsRet, metricsHigh = [], []
    backtestOutcomes, backtestHistories, refsOutcomes = [], [], []
    instStats = {}

    logs2File = ""
    models, bestIters = [], []
    foldNum = -1
    for train, test in self.kFolds:
      foldNum += 1
            
      trainLen = len(train)
      testLen = len(test)
      #datasets preparation for the current train/test fold for highs and returns.
      #train
      X_train, _ = self.featStorage.collectDataWithDatesList(self.dateToXFrames, train, "x", needFullInfo=False)
      Y_train_ret, _ = self.featStorage.collectDataWithDatesList(self.dateToYRetFrames, train, "y", needFullInfo=False)
      Y_train_high, _ = self.featStorage.collectDataWithDatesList(self.dateToYHighFrames, train, "y", needFullInfo=False)
      #test
      X_test, X_info_test = self.featStorage.collectDataWithDatesList(self.dateToXFrames, test, "x")
      Y_test_ret, Y_info_test_ret = self.featStorage.collectDataWithDatesList(self.dateToYRetFrames, test, "y")
      Y_test_high, Y_info_test_high = self.featStorage.collectDataWithDatesList(self.dateToYHighFrames, test, "y")
      #datasets information
      if verb > 0:
        print("=" * 20 + "- FOLD #" + str(foldNum) + " -" + "=" * 20)
        print("Train: ",trainLen, " days \t//\t", len(Y_train_ret), "frames")
        print("Test:  ", testLen, " days\t//\t", len(Y_test_ret), "frames")
        Y_train_levels_count = np.zeros(len(currentParams["retLevels"]) + 2)
        for binNum in Y_train_ret:
          Y_train_levels_count[binNum] += 1
        plt.bar(range(Y_train_levels_count.shape[0]), Y_train_levels_count)
        plt.title("Return bins distribution")
        print("===== Y_train return levels =====")
        for i, val in enumerate([-currentParams["stopLoss"],*currentParams["retLevels"]]):
          WARN = "!!!!!!!!" if Y_train_levels_count[i] == 0 else ""
          print("<", val, "\t: ", int(Y_train_levels_count[i]), WARN)
        print(">\t\t: ", int(Y_train_levels_count[-1]))
        print("x=x=x Y_train return levels x=x=x")
        Y_train_high_levels_count = np.zeros(len(currentParams["highLevels"]) + 1)
        for binNum in Y_train_high:
          Y_train_high_levels_count[binNum] += 1
        print("===== Y_train high levels =====")
        for i, val in enumerate(currentParams["highLevels"]):
          WARN = "!!!!!!!!" if Y_train_high_levels_count[i] == 0 else ""
          print("<",val,"\t: ", int(Y_train_high_levels_count[i]), WARN)
        print(">\t\t: ", Y_train_high_levels_count[-1])
        print("x=x=x Y_train high levels x=x=x")
        
      trainingParams, libSpecificParams = {"ITERS" : currentParams["ITERS"], "CLASSIFIER" : True, "SEED" : currentParams["seed"], "LOSS": "MultiClass"}, {}
     
      models.append([])
      bestIters.append([])

      #training for returns
      Y_preds_ret, modelClass_ret = catBoostTrain(X_train, Y_train_ret, X_test, Y_test_ret, X_test, trainingParams,libSpecificParams)

      #training for highs
      Y_preds_high, modelClass_high = catBoostTrain(X_train, Y_train_high, X_test, Y_test_high, X_test, trainingParams,libSpecificParams)

      #probabilities of pairs (not used)
      Y_pair_probs = np.ones((len(X_test), len(currentParams["highLevels"]) + 1, len(currentParams["retLevels"]) + 1 + 1))

      #calculation of optimal TP and expected results according to predicted probability densities of highs and returns
      predRets, predTPs = self.calcOptimalTPandProfit(currentParams, Y_preds_ret, Y_preds_high, Y_info_test_ret, Y_pair_probs) #with pairs
      expectedReturns.extend(predRets)
      optimalTPs.extend(predTPs)

      metricsRet.append(self.estimateMetrics(Y_test_ret, Y_preds_ret))
      metricsHigh.append(self.estimateMetrics(Y_test_high, Y_preds_high))

      models[-1].append(modelClass_ret)
      models[-1].append(modelClass_high)
      bestIters[-1].append(modelClass_ret.get_best_iteration())
      bestIters[-1].append(modelClass_high.get_best_iteration())
      
      #backtest day-by-day simulation accoring to optimal TPs and expected returns
      backtestOutcome, backtestHistory, backtestRefs, newBuysOutcomes, newDayOutcomes, newExpectedOutcomes, newOptimalTPs, newLogs2File = self.runBacktest(currentParams, test, self.dateToYRetFrames, predRets, predTPs, refs, instStats, logFile=logFile, verb=verb)
      logs2File += newLogs2File

      buysOutcomes.extend(newBuysOutcomes)
      dayOutcomes.extend(newDayOutcomes)
      expectedOutcomes.extend(newExpectedOutcomes)
      optimalTPs.extend(newOptimalTPs)

      print("backtest ret_metrics: \t {:.4f} \t Outcome: {:.3f}%".format(metricsRet[-1], backtestOutcome*100))
      print("backtest high_metrics: \t {:.4f} \t Outcome: {:.3f}% ".format(metricsHigh[-1], backtestOutcome*100))
      refsOutcomes.append([])
      stRefs = ""
      for refOutcome in backtestRefs:
        stRefs += "RefOut: {:.3f}% ".format(refOutcome * 100)
        refsOutcomes[-1].append(refOutcome)
      if stRefs !="": print(stRefs)
      print()
      backtestOutcomes.append(backtestOutcome)
      backtestHistories.append(backtestHistory)
      
    print()

    metricsRet = np.array(metricsRet)
    metricsHigh =  np.array(metricsHigh)

    print("ret_metrics {:.4f}: {:.3f} - {:.3f}".format(metricsRet.mean(), metricsRet.min(), metricsRet.max()))
    #print("ret_metrics {:.4f}: {:.3f}% - {:.3f}".format(metricsRet.prod()**(1/len(metricsRet)), metricsRet.min(), metricsRet.max()))
    print("high_metrics {:.4f}: {:.3f} - {:.3f}".format(metricsHigh.mean(), metricsHigh.min(), metricsHigh.max()))
    #print("high_metrics {:.4f}: {:.3f}% - {:.3f}".format(metricsHigh.prod()**(1/len(metricsHigh)), metricsHigh.min(), metricsHigh.max()))
    plt.figure()
    plt.hist(metricsRet, color='g')
    plt.title("ret_metrics")
    plt.figure()
    plt.hist(metricsHigh, color='r')
    plt.title("high_metrics")

    outcomes = np.array(backtestOutcomes)
    geomMean = np.prod(outcomes) ** (1/outcomes.shape[0])
    outcomesSt = "OUTCOME {:.3f}%: {:.3f}% - {:.3f}%".format(geomMean * 100, outcomes.min() * 100, outcomes.max() * 100)
    print(outcomesSt)
    plt.figure()
    plt.hist(outcomes, color='r', log=True)
    plt.title("Outcomes")

    if len(refsOutcomes) > 0:
      for i in range(len(refsOutcomes[0])):
        outcomes = []
        for j in range(len(refsOutcomes)):
          outcomes.append(refsOutcomes[j][i])
        outcomes = np.array(outcomes)
        geomMean = np.prod(outcomes) ** (1/outcomes.shape[0])
        print("REF OUTCOME {:.3f}%: {:.3f}% - {:.3f}%".format(geomMean * 100, outcomes.min() * 100, outcomes.max() * 100))
        plt.figure()
        plt.hist(outcomes, color='c', log=True)
        plt.title("Ref Outcomes")

    plt.figure(figsize=(13,6))
    plt.hist(np.array(expectedOutcomes)*100, bins=50)
    plt.yscale("log")
    plt.grid()
    plt.title("Expected outcomes, +%")

    plt.figure(figsize=(13,6))
    plt.hist(np.array(optimalTPs)*100, bins=50)
    plt.yscale("log")
    plt.grid()
    plt.title("Optimal TP, +%")

    plt.figure(figsize=(13,6))
    plt.hist(np.array(buysOutcomes)*100, log=True, bins=50)
    plt.grid()
    plt.title("Buys outcomes, +%")

    plt.figure(figsize=(13,6))
    plt.hist(np.array(dayOutcomes)*100, color='r', bins=50)
    plt.grid()
    plt.title("Day outcomes, +%")

    plt.figure(figsize=(13,6))
    plt.hist(np.array(dayOutcomes)*100, color='r', bins=50)
    plt.grid()
    plt.yscale("log")
    plt.title("Day outcomes logy, +%")
        
    plt.figure(figsize=(12,8))
    for backtestHistory in backtestHistories:
      plt.plot(backtestHistory)
    plt.yscale("log")
    plt.grid()

    infoPack = {}
    infoPack["creationTime"] = tinkoff.invest.utils.datetime.now()
    infoPack["params"] = currentParams
    infoPack["models"] = models
    infoPack["bestIters"] = bestIters
    infoPack["outcomeSt"] = outcomesSt
    infoPack["outcomes"] = backtestOutcomes
    #infoPack["featTitles"] = featTitles.copy()
    if debugTrain:
      infoPack["expectedReturns"] = expectedReturns
      infoPack["optimalTPs"] = optimalTPs
      infoPack["retTest"] = Y_test_ret
      infoPack["retPreds"] = Y_preds_ret
      infoPack["highTrain"] = Y_train_high
      infoPack["highTest"] = Y_test_high
      infoPack["highPreds"] = Y_preds_high
      infoPack["Y_test_high"] = Y_test_high

    for inst in instStats:
      instStats[inst]["meanResult"] = instStats[inst]["totalResult"] ** (1/instStats[inst]["boughtTimes"])
    infoPack["instStats"] = instStats
    sortedInstStats = []
    for inst in instStats:
      sortedInstStats.append((instStats[inst]["boughtTimes"], inst, instStats[inst]))
    sortedInstStats = sorted(sortedInstStats, key=lambda x:x[0], reverse=True)
    print()
    print("Mostly used instruments:")
    for i in range(min(5, len(sortedInstStats))):
      print(sortedInstStats[i][1], sortedInstStats[i][2])
    sortedInstStats = []
    for inst in instStats:
      sortedInstStats.append((instStats[inst]["meanResult"], inst, instStats[inst]))
    sortedInstStats = sorted(sortedInstStats, key=lambda x:x[0], reverse=True)
    print()
    print("Best mean result:")
    for i in range(min(5, len(sortedInstStats))):
      print(sortedInstStats[i][1], sortedInstStats[i][2])
    
    sortedInstStats = []
    for inst in instStats:
      sortedInstStats.append((instStats[inst]["totalResult"], inst, instStats[inst]))
    sortedInstStats = sorted(sortedInstStats, key=lambda x:x[0], reverse=False)
    print()
    print("Worst total loss:")
    for i in range(min(5, len(sortedInstStats))):
      print(sortedInstStats[i][1], sortedInstStats[i][2])

    plt.figure(figsize=(10, 6))
    featImportances = []
    for i in range(len(models)):
      featImportances.append(models[i][0].get_feature_importance())
      plt.scatter(range(len(featImportances[i])), featImportances[i], alpha=0.4)
    plt.grid()
    plt.xticks(rotation='vertical');
    plt.title("Return predict feature importance")

    plt.figure(figsize=(10, 6))
    featImportances = []
    for i in range(len(models)):
      featImportances.append(models[i][1].get_feature_importance())
      plt.scatter(range(len(featImportances[i])), featImportances[i], alpha=0.4)
    plt.grid()
    plt.xticks(rotation='vertical');
    plt.title("High predict feature importance")

    
    if logFile:
      with open(logFile, "w") as f:
        f.write(logs2File)

    if saveTo:
      num = 0
      while os.path.exists("./models/"+saveTo + "_" + str(num) + ".pickle"):
        num += 1
      with open("./models/"+saveTo  + "_" +  str(num) + ".pickle", "wb") as f:
        pickle.dump(infoPack, f)
    with open("./tempLastModels.pickle", "wb") as f:
      pickle.dump(infoPack, f)
    print(currentParams)

    return infoPack

  ############################################################################################################

  def runPredict(self, dateToXPredFrames, dates, model_ret, model_high, params={}, stats={}, verb=1):
    if params == {}: params = self.currentParams
    currentParams = params
    if verb > 0:
      print(currentParams)
    X_pred, X_pred_info = self.featStorage.collectDataWithDatesList(dateToXPredFrames, dates, "x")
    Y_preds_ret = model_ret.predict_proba(X_pred)
    Y_preds_high = model_high.predict_proba(X_pred)
    Y_pair_probs = np.ones((len(X_pred), len(currentParams["highLevels"]) + 1, len(currentParams["retLevels"]) + 1 + 1))
    predRets, predTPs = self.calcOptimalTPandProfit(currentParams, Y_preds_ret, Y_preds_high, X_pred_info, Y_pair_probs)
    
    if stats == {}:
      stats["stopLossesAmount"] = 0
      stats["takeProfitsAmount"] = 0
      stats["profitDealsAmount"] = 0
      stats["lossDealsAmount"] = 0
    ind = 0
    indBase = ind
    instructions = []
    history = [1]
    for dateInd in range(len(dates)):
      bank = history[-1]
      stDayOutcome = str(dates[dateInd])[:10]
      buys, ind = self.calcBuysAndBankParts(currentParams, dateToXPredFrames[dates[dateInd]], predRets, predTPs, ind, 1, -2)
      instructionsForToday = []
      for buy in buys:
        buyInfo = buy[1]
        inst = buyInfo["frameInfo"]["inst"]
        instInfo = self.featStorage.instInfo[inst]
        frameInd = buyInfo["frameInfo"]["frameInd"]
        ticker = self.featStorage.instMetainfo[inst].ticker
        buyPrice = instInfo["open"][frameInd] * 1.005
        sellPrice = instInfo["open"][frameInd] * (1+buy[1]["TP"])
        stopLoss = buyPrice * (1 - currentParams["stopLoss"])
        description = "Buy " + ticker + " for {:.2f}, sell for {:.2f}({:+.2f}%)".format(buyPrice, sellPrice, 100*buy[1]["TP"]) + " Bank part {:.2f}".format(buy[1]["bankPart"]) + " Expected result {:+.2f}% Stop Loss {:.2f}({:+.2f}%)".format(100 * buy[0], stopLoss, -100*currentParams["stopLoss"])
        instructionsForToday.append({"ticker" : ticker, "figi" : inst, "buyPrice" : buyPrice, "sellPrice" : sellPrice, "stopLoss" : stopLoss, "bankPart" : buy[1]["bankPart"], "day" : dates[dateInd], "description" : description})

        if dateInd < len(dates) - 1: #if date is not last - it have results
          o, h, l, c = instInfo["open"][frameInd], instInfo["high"][frameInd+1], instInfo["low"][frameInd+1], instInfo["close"][frameInd+1]
          STOP_LOSS = params["stopLoss"]
          TAKE_PROFIT = max(currentParams["minTakeProfit"], buyInfo["TP"])
          
          if (l / o) - 1 < -STOP_LOSS:
            outcome = 1 - STOP_LOSS
            stats["stopLossesAmount"] += 1
            instructionsForToday[-1]["outcomeType"] = "SL"
          elif (h / o) - 1 > TAKE_PROFIT:
            outcome = 1 + TAKE_PROFIT
            stats["takeProfitsAmount"] += 1
            instructionsForToday[-1]["outcomeType"] = "TP"
          else:
            outcome = c / o
            if c > o: stats["profitDealsAmount"] += 1
            else: stats["lossDealsAmount"] += 1
            instructionsForToday[-1]["outcomeType"] = "--"
          instructionsForToday[-1]["outcome"] = outcome
          bankTook = buy[1]["bankPart"] * history[-1]
          bank -= bankTook
          bankTook *= 1 - params["comission"]
          bankGot = bankTook * outcome
          bankGot *= 1 - params["comission"]
          bank += bankGot
        else:
          if verb > 0:
            print(description)
      instructions.extend(instructionsForToday)
      history.append(bank)
      if (verb > 0) and (dateInd < len(dates) - 1):
        stDayOutcome += " {:+.2f}% ({:.3f})".format(100 * (history[-1] / history[-2] - 1), history[-1]) + "\n"
        for instruction in instructionsForToday:
          stDayOutcome += "{:+.2f}%".format(100 * (instruction["outcome"] - 1)) + " " + instruction["outcomeType"] + " " +instruction["description"] + "\n"
        print(stDayOutcome + "\n")
    if (verb > 0) and (len(dates) > 1):
      print("Take Profit amount: \t", stats["takeProfitsAmount"])
      print("Stop Loss amount: \t", stats["stopLossesAmount"])
      print("Profit deals amount: \t", stats["profitDealsAmount"])
      print("Loss deals amount: \t", stats["lossDealsAmount"])
      plt.plot(history)
      print()
      print("Resulting bank -", history[-1])

    return instructions, history

  ############################################################################################################

if __name__ == "__main__":
  pass

