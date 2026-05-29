import requests
url='https://resources.premierleague.com/premierleague/photos/players/250x250/p85971.jpg'
print('Trying without Referer...')
try:
    r=requests.get(url, timeout=10)
    print('status', r.status_code)
    print('content-type', r.headers.get('content-type'))
except Exception as e:
    print('error', e)

print('\nTrying with Referer...')
try:
    r=requests.get(url, headers={'Referer':'https://fantasy.premierleague.com/','User-Agent':'curl/7.79.1'}, timeout=10)
    print('status', r.status_code)
    print('content-type', r.headers.get('content-type'))
except Exception as e:
    print('error', e)
