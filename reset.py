import json
import os
import sys
from pathlib import Path
from typing import Optional

from apps.shared.constants import get_blossom_robot
from apps.shared.models.robot import BlossomRobot

# Windows-specific keyboard input handling
if sys.platform == "win32":
    import msvcrt
else:
    import termios
    import tty

class ResetSequenceEditor:
    def __init__(self):
        self.sequence_file = Path("sequences/reset_sequence.json")
        self.motors = []
        self.current_motor_index = 0
        self.robot: Optional[BlossomRobot] = None
        self.robot_connected = False
        self.load_sequence()
        self.connect_robot()
        
    def load_sequence(self):
        """Load the current reset sequence from JSON file"""
        try:
            with open(self.sequence_file, 'r') as f:
                data = json.load(f)
                # Extract motor positions from the first frame
                positions = data["frame_list"][0]["positions"]
                self.motors = [(pos["dof"], float(pos["pos"])) for pos in positions]
        except FileNotFoundError:
            print(f"Error: {self.sequence_file} not found!")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading sequence: {e}")
            sys.exit(1)
    
    def connect_robot(self):
        """Attempt to connect to the robot"""
        try:
            print("Connecting to robot...")
            self.robot = get_blossom_robot()
            self.robot_connected = True
            print("✓ Robot connected successfully!")
            
            # Get available motor names from robot
            robot_motor_names = [motor.name for motor in self.robot.motors]
            print(f"Available robot motors: {robot_motor_names}")
            
        except Exception as e:
            print(f"⚠ Warning: Could not connect to robot: {e}")
            print("You can still edit the sequence, but won't see real-time movement.")
            self.robot_connected = False
    
    def send_position_to_robot(self, motor_name: str, position: float):
        """Send a single motor position to the robot"""
        if not self.robot_connected or self.robot is None:
            return
            
        try:
            # Convert the sequence position (0-10 range) to robot angle
            # Using the same conversion as in the Sequence class
            converted_position = self._convert_sequence_to_robot_angle(position)
            
            # Apply robot-specific bounds
            if motor_name in self.robot.range_pos:
                min_pos, max_pos = self.robot.range_pos[motor_name]
                converted_position = max(min_pos, min(max_pos, converted_position))
            
            # Send position to robot (quick movement)
            self.robot.goto_position({motor_name: converted_position}, 0.2, wait=False)
            
        except Exception as e:
            print(f"Error sending position to robot: {e}")
    
    def _convert_sequence_to_robot_angle(self, sequence_pos: float) -> float:
        """Convert sequence position (0-10) to robot angle using the same logic as Sequence class"""
        return (sequence_pos - 3) * 50
    
    def save_sequence(self):
        """Save the updated sequence back to JSON file"""
        try:
            # Create the sequence structure
            sequence_data = {
                "animation": "reset",
                "frame_list": [
                    {
                        "millis": 0,
                        "positions": [
                            {"dof": motor_name, "pos": f"{position:.2f}"}
                            for motor_name, position in self.motors
                        ]
                    },
                    {
                        "millis": 10,
                        "positions": [
                            {"dof": motor_name, "pos": f"{position:.2f}"}
                            for motor_name, position in self.motors
                        ]
                    }
                ]
            }
            
            with open(self.sequence_file, 'w') as f:
                json.dump(sequence_data, f, indent=4)
            print(f"\n✓ Sequence saved to {self.sequence_file}")
            
        except Exception as e:
            print(f"Error saving sequence: {e}")
    
    def get_key(self):
        """Get a single keypress (cross-platform)"""
        if sys.platform == "win32":
            # Windows
            key = msvcrt.getch()
            if key == b'\xe0':  # Arrow key prefix on Windows
                key = msvcrt.getch()
                if key == b'H':    # Up arrow
                    return 'up'
                elif key == b'P':  # Down arrow
                    return 'down'
                elif key == b'K':  # Left arrow
                    return 'left'
                elif key == b'M':  # Right arrow
                    return 'right'
            elif key == b'\r':     # Enter
                return 'enter'
            elif key == b'\x1b':   # Escape
                return 'escape'
            else:
                return key.decode('utf-8', errors='ignore')
        else:
            # Unix/Linux/Mac
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                key = sys.stdin.read(1)
                if key == '\x1b':  # Escape sequence
                    key += sys.stdin.read(2)
                    if key == '\x1b[A':    # Up arrow
                        return 'up'
                    elif key == '\x1b[B':  # Down arrow
                        return 'down'
                    elif key == '\x1b[D':  # Left arrow
                        return 'left'
                    elif key == '\x1b[C':  # Right arrow
                        return 'right'
                elif key == '\r' or key == '\n':
                    return 'enter'
                elif key == '\x1b':
                    return 'escape'
                return key
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    def display_interface(self):
        """Display the current state of the editor"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 60)
        print("         ROBOT RESET SEQUENCE EDITOR")
        print("=" * 60)
        
        # Robot connection status
        if self.robot_connected:
            print("🤖 Robot Status: CONNECTED - Real-time movement enabled")
        else:
            print("⚠  Robot Status: DISCONNECTED - Editing sequence only")
        print()
        
        print("Motors and their current positions:")
        print("-" * 50)
        
        for i, (motor_name, position) in enumerate(self.motors):
            # Calculate robot angle for display
            robot_angle = self._convert_sequence_to_robot_angle(position)
            
            if i == self.current_motor_index:
                print(f"► {motor_name:12} : {position:6.2f} (→ {robot_angle:6.1f}°)  ◄ SELECTED")
            else:
                print(f"  {motor_name:12} : {position:6.2f} (→ {robot_angle:6.1f}°)")
        
        print()
        print("-" * 50)
        print("Controls:")
        print("  ← → : Select motor")
        print("  ↑ ↓ : Adjust position (+/- 0.1)")
        print("  S   : Save sequence")
        print("  R   : Reset to robot's default position")
        print("  Q   : Quit without saving")
        print("  ESC : Quit without saving")
        print("-" * 50)
        
        current_motor = self.motors[self.current_motor_index][0]
        current_pos = self.motors[self.current_motor_index][1]
        current_angle = self._convert_sequence_to_robot_angle(current_pos)
        print(f"Current: {current_motor} = {current_pos:.2f} (Robot angle: {current_angle:.1f}°)")
        
        if self.robot_connected:
            print("💡 Tip: Move arrows to see the robot move in real-time!")
    
    def run(self):
        """Main editor loop"""
        print("Loading reset sequence editor...")
        
        while True:
            self.display_interface()
            
            try:
                key = self.get_key().lower()
                
                if key == 'left':
                    # Move to previous motor
                    self.current_motor_index = (self.current_motor_index - 1) % len(self.motors)
                    # Send current position of newly selected motor to robot
                    motor_name, position = self.motors[self.current_motor_index]
                    self.send_position_to_robot(motor_name, position)
                
                elif key == 'right':
                    # Move to next motor
                    self.current_motor_index = (self.current_motor_index + 1) % len(self.motors)
                    # Send current position of newly selected motor to robot
                    motor_name, position = self.motors[self.current_motor_index]
                    self.send_position_to_robot(motor_name, position)
                
                elif key == 'up':
                    # Increase current motor position
                    motor_name, current_pos = self.motors[self.current_motor_index]
                    new_pos = min(10.0, current_pos + 0.1)  # Cap at 10.0
                    self.motors[self.current_motor_index] = (motor_name, new_pos)
                    # Send to robot in real-time
                    self.send_position_to_robot(motor_name, new_pos)
                
                elif key == 'down':
                    # Decrease current motor position
                    motor_name, current_pos = self.motors[self.current_motor_index]
                    new_pos = max(0.0, current_pos - 0.1)  # Cap at 0.0
                    self.motors[self.current_motor_index] = (motor_name, new_pos)
                    # Send to robot in real-time
                    self.send_position_to_robot(motor_name, new_pos)
                
                elif key == 's':
                    # Save sequence
                    self.save_sequence()
                    input("\nPress Enter to continue...")
                
                elif key == 'r':
                    # Reset to robot's default positions
                    if self.robot_connected and self.robot:
                        print("\nResetting to robot's default positions...")
                        for i, (motor_name, _) in enumerate(self.motors):
                            if motor_name in self.robot.reset_pos:
                                # Convert robot angle back to sequence position
                                robot_angle = self.robot.reset_pos[motor_name]
                                sequence_pos = (robot_angle / 50) + 3  # Reverse conversion
                                sequence_pos = max(0.0, min(10.0, sequence_pos))  # Clamp
                                self.motors[i] = (motor_name, sequence_pos)
                                self.send_position_to_robot(motor_name, sequence_pos)
                        input("Reset complete! Press Enter to continue...")
                    else:
                        print("\nRobot not connected - cannot reset to default positions.")
                        input("Press Enter to continue...")
                
                elif key == 'q' or key == 'escape':
                    # Quit
                    print("\nExiting without saving...")
                    if self.robot_connected:
                        print("Robot will remain in current position.")
                    break
                    
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                input("Press Enter to continue...")

def main():
    """Main entry point"""
    print("Starting Reset Sequence Editor...")
    print("Make sure you're in the project root directory!")
    print()
    
    # Check if we're in the right directory
    if not Path("sequences/reset_sequence.json").exists():
        print("Error: sequences/reset_sequence.json not found!")
        print("Please run this script from the project root directory.")
        input("Press Enter to exit...")
        return
    
    editor = ResetSequenceEditor()
    editor.run()

if __name__ == "__main__":
    main()