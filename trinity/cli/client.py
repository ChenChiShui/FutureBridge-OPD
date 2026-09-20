import requests


def send_get_request(url: str, params: dict):
    """
    Send GET request with parameters.

    Args:
        url (str): url of server (e.g., )
        params (str): parameters that need to be passed
    """
    try:
        # send GET request
        response = requests.get(url, params=params)

        # check HTTP status code
        response.raise_for_status()

        # print status code and return response json
        print(f"Status Code: {response.status_code}")
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Failed with: {e}")
        return None


def request(url, **kwargs):
    return send_get_request(url, kwargs)


if __name__ == "__main__":
    # --- only for local testing
    LOCAL_DATA_PROCESSOR_SERVER_URL = ""
    LOCAL_TRINITY_TRAINING_SERVER_URL = ""
    # --- only for local testing

    res = request(
        url=LOCAL_DATA_PROCESSOR_SERVER_URL,
        configPath="examples/grpo_gsm8k/gsm8k.yaml",
    )
    if res:
        print(res)
        print(res["message"])
