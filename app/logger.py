import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Loguri salvate în fișier
        logging.StreamHandler(),  # Loguri afișate și în consolă
    ],
)

logger = logging.getLogger("bursa_app")
