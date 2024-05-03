import platform
import os
#TINKOFF_INVEST_API_TOKEN
#TINKOFF_INVEST_ACC-ID-1
#TINKOFF_INVEST_ACC-ID-2
#APPNAME

def printEnvironHelp():
  currentOS = platform.system().lower()
  if currentOS == "linux" or currentOS == "darwin":
    print("\n\nRun in Bash:\nexport TINKOFF_INVEST_API_TOKEN = <your_token>\nexport TINKOFF_INVEST_ACC-ID-1 = <your_account_id>\n\n")
  elif currentOS == 'windows':
    print("\n\nRun in Cmd:\nset TINKOFF_INVEST_API_TOKEN = ‹your_token>\nset TINKOFF_INVEST_ACC-ID-1 = <your_account_id>\n\n")
  print("Or run in Python:\nos.environ[\"TINKOFF_INVEST_API_TOKEN\"] = <your_token>\nos.environ[\"TINKOFF_INVEST_ACC-ID-1\"] = <your_account_id>\n\n")

def getToken():
  if "TINKOFF_INVEST_API_TOKEN" in os.environ:
    return os.environ["TINKOFF_INVEST_API_TOKEN"]
  else:
    print("!!! Error - set environment variables \"TINKOFF_INVEST_API_TOKEN\" and \"TINKOFF_INVEST_ACC-ID-1\"")
    printEnvironHelp()
    return ""

def getAppname():
  if "TINKOFF_INVEST_APPNAME" in os.environ:
    return os.environ["TINKOFF_INVEST_APPNAME"]
  else:
    return ""

def getAccountIDs():
  accountIDs = []
  if "TINKOFF_INVEST_ACC-ID-1" in os.environ:
    accountIDs.append(os.environ["TINKOFF_INVEST_ACC-ID-1"])
  else:
    print("!!! Error - set environment variables \"TINKOFF_INVEST_API_TOKEN\" and \"TINKOFF_INVEST_ACC-ID-1\"")
    printEnvironHelp()
    return []
  if "TINKOFF_INVEST_ACC-ID-2" in os.environ:
    accountIDs.append(os.environ["TINKOFF_INVEST_ACC-ID-2"])
  return accountIDs



if __name__ == "__main__":
	print(getToken())
	
	
