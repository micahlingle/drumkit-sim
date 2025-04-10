import os
import subprocess
from mido import MidiFile, MidiTrack, Message, second2tick, bpm2tempo

COMMANDS = ["n", "p"]
DRUMS = {
    "kick": 36,
    "snare": 38,
    "rimshot": 40,
    "tom_0": 41,
    "tom_1": 45,
    "tom_2": 48,
    "tom_3": 50,
    "crash_0": 49,
    "crash_1": 57,
    "ride": 51,
    "ride_bell": 53,
    "cowbell": 56,
}
DRUMS_LIST = list(DRUMS.keys())


def play_audio(file_path):
    try:
        subprocess.run(["afplay", file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error playing audio: {e}")
    except FileNotFoundError:
        print("afplay command not found. Ensure it is available on your system.")


def map_drums_to_midi(
    num_drums,
    path_temp,
) -> dict[int, int]:
    """
    Map label i to drum sound using user input

    Input
    - num_drums:
    - path_temp: Path to temp directory where files have already been written for playback

    Output
    - Dictionary mapping drum label to MIDI note value
    """
    drum_to_midi_mapping = {}
    for i in range(num_drums):
        # Get full paths of audio files that we just exported
        wd = f"{path_temp}/{i}/"
        wd_files = os.listdir(wd)
        wd_paths = [os.path.join(wd, file) for file in os.listdir(wd)]
        current_file_index = 0

        print(f"COMMANDS:\n\t{COMMANDS}")
        print(f"DRUMS:\n\t{DRUMS_LIST}")
        user_input = ""
        while user_input not in DRUMS_LIST:
            # Repeat is implicit
            if user_input in COMMANDS:
                if user_input == "n":
                    current_file_index += 1
                elif user_input == "p":
                    current_file_index -= 1

            effective_index = current_file_index % len(wd_files)
            print(f"Playing drum {i}, sample {wd_files[effective_index]}...")
            play_audio(wd_paths[effective_index])

            user_input = input(
                f"""Please
    (1) input a command to hear the same (r), previous (p), or next (n) drum sample again, or
    (2) input a drum name to map detected drum {i} to a MIDI drum: """
            )

        drum_to_midi_mapping[i] = DRUMS[user_input]
        print(f"Set drum {i} to {user_input}.")
    return drum_to_midi_mapping


# 480 ticks per beat is higher standard amount
# https://mido.readthedocs.io/en/stable/files/midi.html#converting-between-ticks-and-seconds
TICKS_PER_BEAT = 480


def write_midi(
    peaks: list,
    labels: list,
    velocities: list,
    drum_to_midi_map: dict,
    sr: int,
    bpm: int,
    time_signature: tuple[int, int],
):
    midi_file = MidiFile()
    midi_file.ticks_per_beat = TICKS_PER_BEAT
    # MIDI tempo is microseconds per quarter note.
    midi_file.tempo = bpm2tempo(bpm, time_signature)

    # Create a track and add it to the MIDI file
    track = MidiTrack()
    midi_file.tracks.append(track)

    # Add messages to the track
    # track.append(Message('program_change', program=12, time=0))
    tick_curr = 0
    tick_prev = 0
    for peak, label, velocity in zip(peaks, labels, velocities):
        tick_curr = second2tick(peak / sr, midi_file.ticks_per_beat, midi_file.tempo)
        delta = tick_curr - tick_prev
        track.append(
            Message(
                "note_on",
                note=drum_to_midi_map[label],
                velocity=velocity,
                time=delta,
            )
        )
        track.append(
            Message(
                "note_off",
                note=drum_to_midi_map[label],
                velocity=velocity,
                time=second2tick(
                    peak / sr + 0.1, midi_file.ticks_per_beat, midi_file.tempo
                ),
            )
        )

        tick_prev = tick_curr

    # Save the MIDI file
    midi_file.save("debug/my.mid")
