# Robot Control API

FastAPI backend for controlling the robot and managing animation sequences.

## Setup

1. Install dependencies:
```bash
pip install fastapi uvicorn
```

2. Run the server:
```bash
# From the project root
uvicorn apps.web.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
```
GET /
```
Returns server status.

### Reset Robot
```
POST /reset
```
Resets the robot to its default position.

**Response:**
```json
{
  "message": "Robot reset successfully"
}
```

### Play Custom Sequence
```
POST /sequences/play
```
Plays a custom animation sequence.

**Request Body:**
```json
{
  "name": "my_animation",
  "frames": "[{\"positions\": [{\"dof\": \"tower_1\", \"pos\": 3}, ...], \"millis\": 0}]"
}
```

**Response:**
```json
{
  "message": "Playing sequence 'my_animation' with 2 frames",
  "sequence_name": "my_animation",
  "frame_count": 2
}
```

## Implementation TODOs

The following endpoints have placeholder implementations that need to be completed:

### Reset Endpoint
```python
@app.post("/reset")
async def reset():
    # TODO: Implement robot reset logic
    # Example:
    # - Connect to robot hardware
    # - Set all motors to default positions
    # - tower_1, tower_2, tower_3, base: 3
    # - ears: 5
    pass
```

### Play Sequence Endpoint
```python
@app.post("/sequences/play")
async def play_custom_sequence(sequence: SequenceResponse):
    # TODO: Implement sequence playback logic
    # Example implementation:
    
    frames = json.loads(sequence.frames)
    
    for frame in frames:
        # 1. Extract positions from frame
        positions = frame["positions"]
        millis = frame["millis"]
        
        # 2. Set each motor position
        for position in positions:
            dof = position["dof"]  # degree of freedom (motor name)
            pos = position["pos"]  # position value (1-5)
            
            # Set motor position
            # robot.set_position(dof, pos)
        
        # 3. Wait for the specified time
        # await asyncio.sleep(millis / 1000)
    
    return {"message": "Sequence complete"}
```

## Sequence Data Format

### Frame Structure
```python
{
    "positions": [
        {
            "dof": "tower_1",  # Motor identifier
            "pos": 3.5         # Position value (1-5)
        },
        # ... more positions
    ],
    "millis": 0  # Timestamp in milliseconds
}
```

### Degrees of Freedom (DOF)
- `tower_1`: First tower motor
- `tower_2`: Second tower motor
- `tower_3`: Third tower motor
- `base`: Base motor
- `ears`: Ears motor

### Position Range
All positions are in the range `1.0` to `5.0`

## CORS Configuration

The API is configured to allow requests from any origin for development:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**For production**, update `allow_origins` to only include your frontend URL:
```python
allow_origins=["http://your-frontend-url.com"],
```

## Testing

### Using curl

**Reset Robot:**
```bash
curl -X POST http://localhost:8000/reset
```

**Play Sequence:**
```bash
curl -X POST http://localhost:8000/sequences/play \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test_animation",
    "frames": "[{\"positions\": [{\"dof\": \"tower_1\", \"pos\": 3}], \"millis\": 0}]"
  }'
```

### Using the API Documentation

FastAPI provides automatic interactive documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Integration with Robot Hardware

To integrate with actual robot hardware, you'll need to:

1. Import your robot control library
2. Initialize the robot connection
3. Implement the position setting logic in the endpoints
4. Handle timing and synchronization for smooth animations

Example integration:
```python
# Import robot library
from your_robot_library import Robot

# Initialize robot
robot = Robot()

@app.post("/reset")
async def reset():
    robot.reset_to_default()
    return {"message": "Robot reset successfully"}

@app.post("/sequences/play")
async def play_custom_sequence(sequence: SequenceResponse):
    frames = json.loads(sequence.frames)
    
    for frame in frames:
        for position in frame["positions"]:
            robot.set_position(position["dof"], position["pos"])
        await asyncio.sleep(frame["millis"] / 1000)
    
    return {"message": "Sequence complete"}
```

