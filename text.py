import requests
import json
import os
from dotenv import load_dotenv


load_dotenv()


api_key = os.getenv("DEEPSEEK_API_KEY")


if not api_key:
    print("Error: DEEPSEEK_API_KEY not found in .env file.")
    print("Please create a .env file and add your API key to it.")
else:
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
        
       
        if response.status_code == 200:
            print(" API key is valid and working!")
            print("Response:", response.json())
        else:
            print(f" API key might be INVALID or has an issue.")
            print(f"Status Code: {response.status_code}")
            print(f"Error Message: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"A network error occurred: {e}")