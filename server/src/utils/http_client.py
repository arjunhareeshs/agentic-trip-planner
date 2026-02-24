import httpx
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

class HttpClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HttpClient, cls).__new__(cls)
            cls._instance.client = httpx.Client(
                timeout=20.0,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
            )
        return cls._instance

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        reraise=True
    )
    def get(self, url: str, params: dict = None, headers: dict = None):
        """
        Perform a GET request with retry logic.
        """
        response = self.client.get(url, params=params, headers=headers)
        if response.status_code >= 500:
            response.raise_for_status()
        return response

    def close(self):
        self.client.close()

# Singleton instance
http_client = HttpClient()
