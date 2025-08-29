import requests
import json

# Replace this with the key you want to test
api_key = "sk-3142ff027f1a43f2bcd5608c181452c3" 

url = "https://api.deepseek.com/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

data = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "user", "content": "Hello"}
    ],
    "max_tokens": 10,
    "stream": False
}

try:
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    # Check the status code
    if response.status_code == 200:
        print("✅ API key is valid and working!")
    else:
        print(f"❌ API key is INVALID or has an issue.")
        print(f"Status Code: {response.status_code}")
        print(f"Error Message: {response.text}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")