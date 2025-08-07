#!/usr/bin/env python3
"""
Examples for the asynchronous aCallTracer class.
"""

import asyncio
import logging

# The synchronous stack() function can still be used
# Assuming aCallTracer is in its own module
from calltracer import aCallTracer, stack, no_self

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

async_trace = aCallTracer(level=logging.DEBUG)
async_chtrace = aCallTracer(level=logging.DEBUG, trace_chain=True, transform=no_self)


class AsyncDataFetcher:
    """An async class to demonstrate tracing concurrent operations"""

    def __init__(self, name):
        self.name = name
        logging.info("Fetcher '%s' initialized", self.name)

    @async_trace
    async def process_item(self, item_id: str, delay: float) -> str:
        """
        A sample async task that simulates I/O operation and gets traced.
        """
        logging.info("-> Starting to process item '%s'", item_id)
        await asyncio.sleep(delay)

        # You can still use the synchronous stack() function inside async code
        if item_id == "B":
            stack(level=logging.INFO)

        logging.info("<- Finished processing item '%s'", item_id)
        return f"Processed {item_id}"

    @async_trace
    async def process_item_medium(self, item_id: str, delay: float) -> str:
        return await self.process_item(item_id, delay)

    @async_trace
    async def process_item_upper(self, item_id: str, delay: float) -> str:
        return await self.process_item_medium(item_id, delay)


class AsyncSecondDataFetcher:
    """An async class to demonstrate tracing concurrent operations with chaining"""

    def __init__(self, name):
        self.name = name
        logging.info("Fetcher '%s' initialized", self.name)

    @async_chtrace
    async def process_item(self, item_id: str, delay: float) -> str:
        """
        A sample async task that simulates I/O operation and gets traced
        """
        await asyncio.sleep(delay)

        return f"Processed {item_id}"

    @async_chtrace
    async def process_item_medium(self, item_id: str, delay: float) -> str:
        return await self.process_item(item_id, delay)

    @async_chtrace
    async def process_item_upper(self, item_id: str, delay: float) -> str:
        return await self.process_item_medium(item_id, delay)


async def main1():
    """Main function to set up and run the async example."""
    fetcher = AsyncDataFetcher("ConcurrentFetcher")

    print("\n--- Running two tasks concurrently to demonstrate task-safety ---")

    # asyncio.gather runs multiple coroutines concurrently.
    # aCallTracer will correctly trace each one without mixing up the logs.
    results = await asyncio.gather(
        fetcher.process_item_upper(item_id="A", delay=0.2),
        fetcher.process_item_upper(item_id="B", delay=0.1),
    )

    logging.info("Concurrent results: %s", results)

    # Now with chaining
    fetcher = AsyncSecondDataFetcher("ConcurrentFetcher2")

    print(
        "\n--- Running two tasks concurrently to demonstrate task-safety one more time ---"
    )

    results = await asyncio.gather(
        fetcher.process_item_upper(item_id="A", delay=0.2),
        fetcher.process_item_upper(item_id="B", delay=0.1),
    )

    logging.info("Concurrent second results: %s", results)


if __name__ == "__main__":
    # Use asyncio.run() to execute the top-level async main() function
    asyncio.run(main1())
