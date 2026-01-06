from time import perf_counter

from loguru import logger


class TimeLogger:
    def __init__(self):
        """
        Log how long the different stages of the reconstruction process take.
        """
        self.tstart = perf_counter()
        self.setup_time_total = 0.0
        self.recon_time_total = 0.0
        self.reporting_time_total = 0.0
        self.total_time = 0.0
        self.setup_time = 0.0
        self.recon_time = 0.0
        self.reporting_time = 0.0

    def setup_start(self):
        self.setup_time = perf_counter()

    def setup_end(self):
        self.setup_time_total += perf_counter() - self.setup_time

    def recon_start(self):
        self.recon_time = perf_counter()

    def recon_end(self):
        self.recon_time_total += perf_counter() - self.recon_time

    def reporting_start(self):
        self.reporting_time = perf_counter()

    def reporting_end(self):
        self.reporting_time_total += perf_counter() - self.reporting_time

    def total_end(self):
        self.total_time = perf_counter() - self.tstart

    def report(self):
        self.total_end()
        overhead_time = self.total_time - (
            self.setup_time_total + self.recon_time_total + self.reporting_time_total
        )

        logger.info(f"Time summary: ")
        logger.info(f"  Total: {self.total_time:.2f} s")
        logger.info(f"    Setup: {self.setup_time_total:.2f} s")
        logger.info(f"    Recon: {self.recon_time_total:.2f} s")
        logger.info(f"    Logging: {self.reporting_time_total:.2f} s")
        logger.info(f"    Overheads: {overhead_time:.2f} s")
