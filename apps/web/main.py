import json
import os
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from apps.shared.models.robot import BlossomRobot
from apps.shared.utils.sequence import get_sequence_by_name
from apps.web.sequence import SequenceRequest
from apps.shared.models.sequence import Sequence
from apps.shared.constants import get_blossom_robot, SEQUENCE_DIR

app = FastAPI()

# Lazy robot initialization to avoid issues with multiprocessing on Windows
# Serial ports can't be shared between processes, so we initialize on-demand
_robot: BlossomRobot | None = None
_robot_error: str | None = None

def get_robot() -> BlossomRobot | None:
    """Get the robot instance, initializing it lazily if needed."""
    global _robot, _robot_error
    if _robot is None and _robot_error is None:
        try:
            _robot = get_blossom_robot()
        except Exception as e:
            _robot_error = str(e)
            print(f"⚠️  Warning: Could not initialize robot: {_robot_error}")
            print("   Running in simulation mode. Robot-dependent endpoints will return simulation responses.")
    return _robot

running_sequences: Dict[str, Sequence] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
async def handle_root():
    robot = get_robot()
    return {
        "message": "Blossom Robot Control API",
        "version": "1.0",
        "robot_connected": robot is not None,
        "mode": "physical" if robot else "simulation",
        "status": "Ready" if robot else "Simulation Mode - No Robot Connected",
    }


@app.post("/reset")
async def handle_reset():
    """Reset the robot to its default position"""
    try:
        robot = get_robot()
        if robot is None:
            return {
                "message": "Reset command received (simulation mode - no physical robot connected)",
                "simulation": True,
            }

        sequence = get_sequence_by_name("reset")
        if sequence is None:
            raise HTTPException(status_code=404, detail="Reset sequence not found")

        sequence.start()
        sequence.wait_to_stop()
        return {"message": "Robot reset successfully", "simulation": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset robot: {str(e)}")


def _read_all_sequence_files() -> list[dict]:
    """Read all sequence JSON files directly from disk (no robot dependency)."""
    sequences = []
    if not os.path.isdir(SEQUENCE_DIR):
        return sequences
    for filename in os.listdir(SEQUENCE_DIR):
        if filename.endswith("sequence.json"):
            filepath = os.path.join(SEQUENCE_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "animation" in data and "frame_list" in data:
                    sequences.append(data)
            except Exception:
                continue
    return sequences


def _find_sequence_file(animation_name: str) -> dict | None:
    """Find and return a single sequence JSON by its animation name."""
    for seq in _read_all_sequence_files():
        if seq["animation"] == animation_name:
            return seq
    return None


@app.get("/sequences")
async def handle_get_sequences():
    """Return all available sequences with metadata."""
    sequences = _read_all_sequence_files()
    return [
        {
            "animation": seq["animation"],
            "frame_count": len(seq.get("frame_list", [])),
            "frame_list": seq.get("frame_list", []),
        }
        for seq in sequences
    ]


# NOTE: /sequences/names must be defined BEFORE /sequences/{sequence_id}
# so FastAPI doesn't interpret "names" as a path parameter.
@app.get("/sequences/names")
async def handle_get_sequence_names():
    """Return a list of all sequence animation names."""
    sequences = _read_all_sequence_files()
    names = sorted(set(seq["animation"] for seq in sequences))
    return {"names": names}


@app.get("/sequences/{sequence_id}")
async def handle_get_sequence(sequence_id: str):
    """Return full data for a specific sequence by animation name."""
    seq = _find_sequence_file(sequence_id)
    if seq is None:
        raise HTTPException(status_code=404, detail=f"Sequence '{sequence_id}' not found")
    return seq


@app.post("/sequences")
async def handle_create_sequence(sequence: SequenceRequest):
    """Save a new sequence to the sequences directory."""
    try:
        os.makedirs(SEQUENCE_DIR, exist_ok=True)
        filename = f"{sequence.animation}_sequence.json"
        filepath = os.path.join(SEQUENCE_DIR, filename)

        seq_data = {
            "animation": sequence.animation,
            "frame_list": [
                {
                    "millis": frame.millis,
                    "positions": [
                        {"dof": pos.dof, "pos": pos.pos}
                        for pos in frame.positions
                    ],
                }
                for frame in sequence.frame_list
            ],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(seq_data, f, indent=4)

        return {
            "message": f"Sequence '{sequence.animation}' saved successfully",
            "animation": sequence.animation,
            "file": filename,
            "frame_count": len(sequence.frame_list),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save sequence: {str(e)}"
        )


@app.get("/sequences/{sequence_id}/play")
async def handle_play_sequence(sequence_id: str):
    """Play a saved sequence by animation name."""
    try:
        robot = get_robot()
        if robot is None:
            # Check if the sequence exists on disk first
            seq_data = _find_sequence_file(sequence_id)
            if seq_data is None:
                raise HTTPException(
                    status_code=404, detail=f"Sequence '{sequence_id}' not found"
                )
            return {
                "message": f"Sequence '{sequence_id}' would be played (simulation mode - no physical robot connected)",
                "animation": sequence_id,
                "frame_count": len(seq_data.get("frame_list", [])),
                "simulation": True,
            }

        # Check if the sequence exists on disk first
        seq_data = _find_sequence_file(sequence_id)
        if seq_data is None:
            raise HTTPException(
                status_code=404, detail=f"Sequence '{sequence_id}' not found"
            )

        sequence = get_sequence_by_name(sequence_id)
        if sequence is None:
            raise HTTPException(
                status_code=404, detail=f"Sequence '{sequence_id}' not found"
            )

        running_sequences[sequence_id] = sequence
        try:
            sequence.start()
            sequence.wait_to_stop()
        finally:
            running_sequences.pop(sequence_id, None)

        return {
            "message": f"Sequence '{sequence_id}' completed successfully",
            "animation": sequence_id,
            "frame_count": len(seq_data.get("frame_list", [])),
            "simulation": False,
        }
    except HTTPException:
        raise
    except Exception as e:
        running_sequences.pop(sequence_id, None)
        raise HTTPException(
            status_code=500, detail=f"Failed to play sequence: {str(e)}"
        )


@app.get("/sequences/{sequence_id}/stop")
async def handle_stop_sequence(sequence_id: str):
    """Stop a currently running sequence."""
    sequence = running_sequences.get(sequence_id)
    if sequence is None:
        raise HTTPException(
            status_code=404,
            detail=f"No running sequence '{sequence_id}' found",
        )
    try:
        sequence.stop()
        running_sequences.pop(sequence_id, None)
        return {
            "message": f"Sequence '{sequence_id}' stopped",
            "animation": sequence_id,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to stop sequence: {str(e)}"
        )


@app.post("/sequences/play")
async def handle_play_custom_sequence(sequence_data: SequenceRequest):
    """
    Play a custom animation sequence.

    The sequence is validated against the Sequence model and then executed on the robot.
    """
    try:
        robot = get_robot()
        if robot is None:
            return {
                "message": f"Sequence '{sequence_data.animation}' would be played (simulation mode - no physical robot connected)",
                "animation": sequence_data.animation,
                "frame_count": len(sequence_data.frame_list),
                "duration_ms": (
                    sequence_data.frame_list[-1].millis if sequence_data.frame_list else 0
                ),
                "simulation": True,
            }

        sequence = Sequence(
            robot=robot,
            animation=sequence_data.animation,
            frames=sequence_data.frame_list,
        )

        print(f"Starting playback of sequence: {sequence_data.animation}")
        sequence.start()
        sequence.wait_to_stop()

        return {
            "message": f"Sequence '{sequence_data.animation}' completed successfully",
            "animation": sequence_data.animation,
            "frame_count": len(sequence_data.frame_list),
            "duration_ms": (
                sequence_data.frame_list[-1].millis if sequence_data.frame_list else 0
            ),
            "simulation": False,
        }

    except ValidationError as e:
        raise HTTPException(
            status_code=400, detail={"error": "Validation error", "details": e.errors()}
        )
    except Exception as e:
        print(f"Error playing sequence: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Failed to play sequence: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Blossom Robot Control API...")
    print("🔌 Attempting robot connection...")

    print("\n🌐 Starting FastAPI server...")
    print("📍 API available at: http://localhost:8000")
    print("📚 API docs available at: http://localhost:8000/docs")
    print("🔄 Auto-reload enabled for development")
    print("\n" + "=" * 50)

    uvicorn.run(
        "apps.web.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
