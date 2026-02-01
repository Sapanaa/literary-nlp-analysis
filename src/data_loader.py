import requests
from bs4 import BeautifulSoup
from pathlib import Path


MOBY_DICK_URL = (
    "https://s3.amazonaws.com/assets.datacamp.com/"
    "production/project_147/datasets/2701-h.htm"
)


def download_moby_dick(save_path: str) -> str:
    """
    Download Moby-Dick HTML, extract clean text,
    and save it locally.
    """
    response = requests.get(MOBY_DICK_URL)
    response.encoding = "utf-8"

    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)

    return text
