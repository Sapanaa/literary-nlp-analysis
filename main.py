from src.data_loader import download_moby_dick

RAW_TEXT_PATH = "data/raw/moby_dick.txt"

if __name__ == "__main__":
    text = download_moby_dick(RAW_TEXT_PATH)
    print("Moby-Dick downloaded.")
    print(f"Number of characters: {len(text)}")
