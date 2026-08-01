import threading
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from operator.transmitter import Transmitter
from operator.visualizer import Visualizer


def run_transmitter():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    transmitter = Transmitter()
    loop.run_until_complete(transmitter.run())


def run_visualizer():
    visualizer = Visualizer()
    visualizer.start()


def main():
    print("[Operator] Starting Transmitter and Visualizer...")
    
    transmitter_thread = threading.Thread(target=run_transmitter, daemon=True)
    transmitter_thread.start()

    run_visualizer()


if __name__ == "__main__":
    main()