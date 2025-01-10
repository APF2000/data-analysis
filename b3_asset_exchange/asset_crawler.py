import requests

# Define the stock ticker and the file name for the JSON response
stock_ticker = 'your_stock_ticker_here'  # Replace with your stock ticker
json_response_file_name = 'your_response_file_name_here.json'  # Replace with desired file name

# URL for the API endpoint
url = 'https://statusinvest.com.br/acao/indicatorhistoricallist'

# Headers to mimic the browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'Origin': 'https://statusinvest.com.br',
    'Connection': 'keep-alive',
    'Referer': f'https://statusinvest.com.br/acoes/{stock_ticker}',
    'Cookie': 'suno_checkout_userid=5853c4e1-14b9-4ee5-b629-004480ed874f; _gcl_au=1.1.1151585502.1711989921; _ga_69GS6KP6TJ=GS1.1.1712839534.7.1.1712839535.59.0.0; _ga=GA1.1.2036242070.1711989921; _adasys=dd169bc7-40d3-480a-95bf-24b44550e6f1; cf_clearance=0gohdl_1ibCnl8tdh4KtPmkJWy_qWrGw0IgogUR8RbE-1712839534-1.0.1.1-ehDsO.F6.vy4KPNaXhsISMYk9hmckqnioppiYdEvEQuIMmOTeAyBXvkKgOVd7NOK4Bezd59XWFXFzpKIl0p_UA; _clck=3ifihk%7C2%7Cfku%7C0%7C1552; _hjSessionUser_1931042=eyJpZCI6ImNiZTQyZjMxLWEwMWEtNWQwMC1iNzRmLWIwN2EyNWU3NjJlNiIsImNyZWF0ZWQiOjE3MTE5ODk5MzE3ODUsImV4aXN0aW5nIjp0cnVlfQ==; __hstc=176625274.21a830d8b7eb3fe45f01e165c4e94ccf.1711989945976.1712758120293.1712837005577.6; hubspotutk=21a830d8b7eb3fe45f01e165c4e94ccf; messagesUtk=9cd758327dcd43c2bcc3594f3404ec1c; G_ENABLED_IDPS=google; _clsk=uhq43b%7C1712839536227%7C4%7C0%7Cj.clarity.ms%2Fcollect; denakop_freq={}; __hssrc=1; _hjSession_1931042=eyJpZCI6IjBkNzA2NzBhLTMzNWUtNDZjZi1hYTNhLWU4OTg1MmNmNWJjZSIsImMiOjE3MTI4MzY5OTg4NDYsInMiOjAsInIiOjAsInNiIjowLCJzciI6MCwic2UiOjAsImZzIjowLCJzcCI6MX0=',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'TE': 'trailers'
}

# Payload for the POST request
data = {
    'codes[]': stock_ticker,
    'time': '5',
    'byQuarter': 'false',
    'futureData': 'false'
}

# Make the POST request to fetch the data
response = requests.post(url, headers=headers, data=data)

# Check if the request was successful
if response.status_code == 200:
    # Write the JSON response to a file
    with open(json_response_file_name, 'w') as file:
        file.write(response.text)
    print(f"Data saved to {json_response_file_name}")
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")
