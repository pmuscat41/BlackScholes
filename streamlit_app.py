# --- BlackScholes.py -------------------------------------------------
from math import log, sqrt, exp
from scipy.stats import norm


class BlackScholes:
    """
    Vanilla-option pricer with Greeks.
    • Handles the special case T = 0  (returns intrinsic value).
    """

    def __init__(
        self,
        current_price: float,
        strike: float,
        time_to_maturity: float,   # in years
        volatility: float,         # annualised, 0-1
        interest_rate: float,      # risk-free, 0-1
    ):
        self.S = current_price
        self.K = strike
        self.T = time_to_maturity
        self.sigma = volatility
        self.r = interest_rate

        # outputs
        self.call_price = 0.0
        self.put_price = 0.0
        self.call_delta = 0.0
        self.put_delta = 0.0
        self.call_gamma = 0.0
        self.put_gamma = 0.0

    # -----------------------------------------------------------------
    def _d1_d2(self):
        d1 = (
            log(self.S / self.K)
            + (self.r + 0.5 * self.sigma**2) * self.T
        ) / (self.sigma * sqrt(self.T))
        return d1, d1 - self.sigma * sqrt(self.T)

    # -----------------------------------------------------------------
    def calculate_prices(self):
        """
        Returns (call_price, put_price) and populates Greeks.
        """
        # ---------- expiry: intrinsic value only ---------------------
        if self.T <= 0.0:
            self.call_price = max(0.0, self.S - self.K)
            self.put_price = max(0.0, self.K - self.S)

            self.call_delta = 1.0 if self.S > self.K else 0.0
            self.put_delta = -1.0 if self.S < self.K else 0.0
            self.call_gamma = self.put_gamma = 0.0
            return self.call_price, self.put_price

        # ---------- standard Black–Scholes ---------------------------
        d1, d2 = self._d1_d2()
        Nd1, Nd2 = norm.cdf(d1), norm.cdf(d2)
        Nmd1, Nmd2 = norm.cdf(-d1), norm.cdf(-d2)

        disc = exp(-self.r * self.T)
        self.call_price = self.S * Nd1 - self.K * disc * Nd2
        self.put_price = self.K * disc * Nmd2 - self.S * Nmd1

        self.call_delta = Nd1
        self.put_delta = Nd1 - 1.0
        self.call_gamma = self.put_gamma = (
            norm.pdf(d1) / (self.S * self.sigma * sqrt(self.T))
        )
        return self.call_price, self.put_price
