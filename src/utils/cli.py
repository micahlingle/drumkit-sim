import argparse as ap
import os


class CliArgs:
    def __init__(self, path, bpm, time_signature, num_drums, debug):
        self.path = path
        self.bpm = bpm
        self.time_signature = time_signature
        self.num_drums = num_drums
        self.debug = debug


cli_args = None


def parse_args():
    global cli_args
    parser = ap.ArgumentParser()
    parser.add_argument(
        "--path",
        default="./datasets/3sounds.wav",
        help="Path to file which will be analyzed",
    )
    parser.add_argument("--bpm", default=60, help="Beats per minute")
    parser.add_argument(
        "--time_signature",
        default="4/4",
        help="Time signature. Only denominator really matters",
    )
    parser.add_argument(
        "--num_drums", type=int, default=3, help="Number of drums in the recording"
    )
    parser.add_argument("--debug", default=False, help="")

    args = parser.parse_args()
    path = args.path
    bpm = int(args.bpm)
    times = args.time_signature.split("/")
    time_signature = (int(times[0]), int(times[1]))
    num_drums = int(args.num_drums)
    debug = args.debug
    if debug:
        os.makedirs("debug", exist_ok=True)

    cli_args = CliArgs(path, bpm, time_signature, num_drums, debug)
