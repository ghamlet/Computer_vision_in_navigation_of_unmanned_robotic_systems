import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ground_station.transmitter import Transmitter
from ground_station.visualizer import Visualizer


async def run_transmitter():
    transmitter = Transmitter()
    await transmitter.run()


async def run_visualizer():
    visualizer = Visualizer()
    await visualizer.start()


async def main():
    print("[Operator] Starting Transmitter and Visualizer...")
    
    # Run both in the same event loop
    await asyncio.gather(
        run_transmitter(),
        run_visualizer(),
    )


if __name__ == "__main__":
    asyncio.run(main())