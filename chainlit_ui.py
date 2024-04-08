import chainlit as cl
import httpx


@cl.on_message
@cl.step
async def main(message: cl.Message):
    request_body = {"sentence": str(message.content)}
    api_url = "http://127.0.0.1:3300/req"

    async with httpx.AsyncClient(timeout=180000) as client:
        headers = {"Content-Type": "application/json"}

        response = await client.post(api_url, json=request_body, headers=headers)

        if response.status_code == 200:
            try:
                json_response = response.json()
                result = json_response.get('result')  # Get 'result' field from JSON response
                if result:
                    await cl.Message(content=result).send()
                else:
                    print("No 'result' field found in JSON response.")
            except Exception as e:
                print(f"Error parsing JSON response: {e}")
        else:
            print(f"Request failed with status code {response.status_code}")
            print(response.text)