import logging
from configparser import ConfigParser
from okx_api.rest_client import OKXRestClient

# Load configuration
config = ConfigParser()
config.read('config/config.ini')

LOG_LEVEL = config.get('GENERAL', 'LOG_LEVEL', fallback='INFO').upper()
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

class Account:
    def __init__(self, simulated_trading: bool = config.getboolean('OKX', 'SIMULATED_TRADING', fallback=False)):
        self.client = OKXRestClient(simulated_trading=simulated_trading)
        self.balance = {}
        self.positions = []

    def fetch_balance(self):
        try:
            response = self.client.get_balance()
            self.balance = response.get('data', [])
            logger.info("Fetched account balance successfully.")
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")

    def fetch_positions(self):
        try:
            response = self.client.get_positions()
            self.positions = response.get('data', [])
            logger.info("Fetched account positions successfully.")
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")

    def get_balance(self):
        return self.balance

    def get_positions(self):
        return self.positions

    def display_balance(self):
        if not self.balance:
            logger.warning("No balance data available. Fetching balance.")
            self.fetch_balance()
        for asset in self.balance:
            logger.info(f"Asset: {asset['ccy']}, Available: {asset['availBal']}, Frozen: {asset['frozenBal']}")

    def display_positions(self):
        if not self.positions:
            logger.warning("No positions data available. Fetching positions.")
            self.fetch_positions()
        for pos in self.positions:
            logger.info(f"Instrument: {pos['instId']}, Quantity: {pos['pos']}, Avg. Price: {pos['avgPx']}")

# Example usage
if __name__ == "__main__":
    account = Account()
    account.display_balance()
    account.display_positions()
