# Blockly Robot Control Frontend

This frontend provides a visual block-based interface for creating custom robot animation sequences.

## Features

- **Visual Block Programming**: Create animation sequences using drag-and-drop blocks
- **Real-time JSON Preview**: See the generated JSON in real-time
- **Robot Control**: Reset and play sequences directly from the interface

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm start
```

The frontend will be available at `http://localhost:8080`

## Using the Interface

### Creating a Sequence

1. **Create Sequence Block**: Drag the "Create sequence named" block to the workspace
   - Set the animation name (e.g., "my_animation")

2. **Add Frames**: Inside the sequence, add "Frame at X milliseconds" blocks
   - Each frame represents a robot position at a specific time
   - Set the milliseconds value for timing

3. **Set Positions**: Inside each frame, add "Set all positions" blocks
   - Configure positions for: tower_1, tower_2, tower_3, base, and ears
   - Values range from 1 to 5

### Example Sequence

```
Create sequence named "wave"
  ├─ Frame at 0 milliseconds
  │   └─ Set all positions (tower_1: 3, tower_2: 3, tower_3: 3, base: 3, ears: 5)
  └─ Frame at 1000 milliseconds
      └─ Set all positions (tower_1: 5, tower_2: 5, tower_3: 3, base: 3, ears: 5)
```

## Control Buttons

### Reset Robot
- Resets the robot to its default position
- Calls the `/reset` endpoint on the backend

### Play Sequence
- Plays the currently created sequence
- Sends the animation data to the backend
- The backend will execute the sequence frame by frame

## Generated JSON Format

The interface generates JSON in the following format:

```json
{
  "animation": "my_animation",
  "frame_list": [
    {
      "positions": [
        {"dof": "tower_1", "pos": 3},
        {"dof": "tower_2", "pos": 3},
        {"dof": "tower_3", "pos": 3},
        {"dof": "base", "pos": 3},
        {"dof": "ears", "pos": 5}
      ],
      "millis": 0
    }
  ]
}
```

This JSON can be copied and saved to a `.json` file for later use.

## API Configuration

The frontend connects to the FastAPI backend at `http://localhost:8000` by default.

To change this, edit the `API_BASE_URL` constant in `src/index.ts`:

```typescript
const API_BASE_URL = 'http://your-api-url:port';
```

## Troubleshooting

### "Error connecting to robot"
- Make sure the FastAPI backend is running
- Check that the API URL is correct
- Ensure CORS is properly configured on the backend

### Sequence not playing
- Verify that you have at least one frame in your sequence
- Check the browser console for error messages
- Ensure all position values are within the valid range (1-5)

