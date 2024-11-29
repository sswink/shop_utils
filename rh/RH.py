import requests
from utils import config_util as cfg

from core.shop_db import Authorize_Tb

cfg.load_config()


class RH:

    def __init__(self, workflowId):
        self.host = cfg.config["host"]
        self.apiKey = cfg.config["apiKey"]
        self.workflowId = workflowId
        self.db = Authorize_Tb()
        self.db.init_tb()

    def submitTask(self, nodeInfoList):
        url = self.host + '/task/openapi/create'
        requestObj = {
            "workflowId": self.workflowId,
            "apiKey": self.apiKey,
            "nodeInfoList": nodeInfoList
        }
        response = requests.post(url, json=requestObj)
        print(response)

    def getResult(self):
        url = self.host + '/task/openapi/outputs'

if __name__ == '__main__':
    rh = RH(workflowId='123')
    rh.db.saveRecord(rh.workflowId,'Test',{"123":"123"},'PROCESSING')
