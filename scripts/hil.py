import sys
import multiprocessing

# If running headlessly (e.g., via SSH), mock pynput so it doesn't crash trying to find an X server.
# This allows smoke-testing the autonomous execution loop.
if "--headless" in sys.argv:
    from unittest.mock import MagicMock
    sys.modules['pynput'] = MagicMock()
    sys.modules['pynput.keyboard'] = MagicMock()

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

from isaaclab.app import AppLauncher

from .utils import common
from .utils.parser import setup_hil_parser
from .utils.common import launch_app_from_args
from lehome.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main entry point for Human-in-the-Loop (HIL) script."""
    parser = setup_hil_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    
    simulation_app = launch_app_from_args(args)
    
    try:
        import lehome.tasks.bedroom
        from .utils.hil_utils import hil_eval

        if getattr(args, "headless", False):
            # In headless mode, pynput cannot connect to X server
            import os
            os.environ["LEHOME_DISABLE_KEYBOARD"] = "1"
            logger.warning("Running in HEADLESS mode. Keyboard teleoperation and HIL hotkeys will NOT work.")
            
        hil_eval(args, simulation_app)
    except Exception as e:
        logger.error(f"Error during HIL evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        common.close_app(simulation_app)


if __name__ == "__main__":
    main()
