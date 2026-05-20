from typing import List
from ..models.sequence import Sequence


def get_all_sequences() -> List[Sequence]:
    return Sequence.get_all_sequences()


def get_all_sequences_str() -> str:
    sequences = get_all_sequences()
    sequence_names = [seq.animation for seq in sequences]
    return "\n".join(sequence_names)


def get_sequence_by_name(name: str) -> Sequence | None:
    target = name.strip().lower()

    for sequence in get_all_sequences():
        animation = sequence.animation.strip().lower()

        if animation == target:
            return sequence

        if f"{animation}_sequence" == target:
            return sequence

        if animation == target.removesuffix("_sequence"):
            return sequence

    print("Loaded sequences:", [seq.animation for seq in get_all_sequences()])
    return None
