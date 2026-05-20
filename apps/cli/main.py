import random
from pypot.utils.stoppablethread import StoppableThread

from apps.shared.utils.sequence import get_all_sequences
from apps.shared.models.sequence import Sequence


def handle_random_sequence_play() -> StoppableThread:
    sequence_list = get_all_sequences()
    number = random.randint(0, len(sequence_list) - 1)
    sequence = sequence_list[number]
    print(f"Playing sequence: {sequence.animation}\n")
    sequence.start()
    return sequence


def print_sequences():
    sequences = get_all_sequences()
    print(Sequence.sequence_str(sequences))


def handle_sequence_play(
    sequence_name: str
) -> StoppableThread | None:
    sequence_list = get_all_sequences()
    for sequence in sequence_list:
        if sequence.animation == sequence_name:
            print(f"Playing sequence: {sequence.animation}\n")
            sequence.start()
            return sequence
    print(f"Sequence {sequence_name} not found\n")
    return None


def main():
    prompt = "(l)ist sequences / (s)equence play / (q)uit: \n"
    active_sequence_thread = None

    while True:
        command = input(prompt)

        if active_sequence_thread is not None and not active_sequence_thread.is_alive():
            active_sequence_thread = None

        match command:
            case "l":
                print_sequences()
            case "s":
                if active_sequence_thread is None:
                    active_sequence_thread = handle_random_sequence_play()
                else:
                    print("Sequence already playing")
            case "q":
                return
            case _:
                if active_sequence_thread is None:
                    active_sequence_thread = handle_sequence_play(command)
                else:
                    print("Sequence already playing")



if __name__ == "__main__":
    main()
