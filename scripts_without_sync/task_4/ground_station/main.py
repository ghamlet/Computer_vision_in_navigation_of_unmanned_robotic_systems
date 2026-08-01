import asyncio
import sys
import os
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ground_station.transmitter import Transmitter
from ground_station.visualizer import Visualizer


def run_visualizer():
    """Run visualizer in a separate thread (it has its own event loop)."""
    visualizer = Visualizer()
    visualizer.start()


async def run_transmitter():
    """Run transmitter in the async event loop."""
    transmitter = Transmitter()
    await transmitter.run()


async def main():
    print("[Operator] Starting Transmitter and Visualizer...")
    
    # Start visualizer in a separate thread
    visualizer_thread = threading.Thread(target=run_visualizer, daemon=True)
    visualizer_thread.start()
    
    # Give visualizer time to start its server
    await asyncio.sleep(2)
    
    # Run transmitter in main async loop
    try:
        await run_transmitter()
    except KeyboardInterrupt:
        print("[Operator] Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())