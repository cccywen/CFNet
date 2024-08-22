import json
import requests

# 不用files用json格式也行的，方法不唯一。只是json的图片数据要多一个base64编码的操作。
r = requests.post('https://xmutpriu.com/api/mathpix_latex?token=bjvM@YqSrh',
                  files={'image': open('C:/Users/Administrator/Desktop/f.png', 'rb')})
d = json.loads(r.text)
print(d['latex'])