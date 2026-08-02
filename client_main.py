import os
import time
import json
import queue
import pprint
import threading
import shutil
import requests
import numpy as np
import sounddevice as sd
import soundfile as sf
import keyboard
import xpc

# Define endpoints and settings
URL = "http://127.0.0.1:8000/process"
AUDIO_FILE_INPUT = "pilot_request.wav"
AUDIO_FILE_OUTPUT = "atc_response.wav"

# Audio recording configuration
CHANNELS = 1
CHUNK = 1024

# Setup keyboard availability flag
KEYBOARD_AVAILABLE = True
try:
    keyboard.is_pressed('space')
except Exception as e:
    print(f"Warning: Keyboard library is not fully operational in this terminal session ({e}).")
    print("Falling back to ENTER-key mode (Press Enter to start, and Enter to stop).")
    KEYBOARD_AVAILABLE = False


def fetch_xplane_telemetry():
    """Connects to local X-Plane 11 via UDP and fetches latitude, longitude, altitude, and heading."""
    print("Fetching live telemetry from X-Plane...")
    telemetry = {
        "latitude": None,
        "longitude": None,
        "altitude": None,
        "heading": None,
        "pitch": None,
        "roll": None,
        "speed": None
    }
    try:
        # Default XPC port is 49009, with a timeout of 1000 milliseconds
        with xpc.XPlaneConnect(timeout=1000) as client:
            pos = client.getPOSI(0)  # 0 is the index for the user's aircraft
            
            try:
                # Fetch indicated airspeed
                speed_dref = client.getDREF("sim/flightmodel/position/indicated_airspeed")
                if speed_dref:
                    telemetry["speed"] = speed_dref[0]
            except Exception as e:
                print(f"Warning: Could not fetch airspeed ({e})")
                
            if pos and len(pos) >= 6:
                telemetry["latitude"] = pos[0]
                telemetry["longitude"] = pos[1]
                telemetry["altitude"] = pos[2]  # Altitude (MSL) in meters
                telemetry["pitch"] = pos[3]     # Pitch in degrees
                telemetry["roll"] = pos[4]      # Roll in degrees
                telemetry["heading"] = pos[5]   # True heading in degrees
                
                speed_str = f", Spd={telemetry['speed']:.1f}kt" if telemetry["speed"] is not None else ""
                print(f"Telemetry successfully fetched: Lat={pos[0]:.5f}, Lon={pos[1]:.5f}, Alt={pos[2]:.1f}m, Hdg={pos[5]:.1f}°{speed_str}")
            else:
                print("Warning: Received empty/malformed position array from X-Plane.")
    except Exception as e:
        print(f"Warning: Could not retrieve X-Plane telemetry ({e}). Simulator may be closed, paused, or network restricted.")
    
    return telemetry


def transmit_audio_and_telemetry(audio_path, telemetry):
    """Sends the audio file and JSON telemetry string as a multipart POST request."""
    print("Transmitting request to AWS AI ATC backend...")
    
    if not os.path.exists(audio_path):
        print(f"Error: Recording file {audio_path} not found.")
        return None

    try:
        with open(audio_path, 'rb') as f:
            files = {
                'audio': ('pilot_request.wav', f, 'audio/wav')
            }
            # Telemetry data sent as JSON string in data parameter
            data = {
                'data': json.dumps(telemetry)
            }
            
            response = requests.post(URL, files=files, data=data)
            print(f"Server Response Status: {response.status_code}")
            
            response.raise_for_status()
            
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                try:
                    json_resp = response.json()
                    print("Received JSON message:")
                    pprint.pprint(json_resp)
                    return None
                except Exception as ex:
                    print(f"Failed to parse JSON response: {ex}")
                    return None
            else:
                # Save binary stream to file
                with open(AUDIO_FILE_OUTPUT, "wb") as out_f:
                    out_f.write(response.content)
                print(f"Successfully saved ATC response to {AUDIO_FILE_OUTPUT} ({len(response.content)} bytes)")
                return AUDIO_FILE_OUTPUT
                
    except requests.exceptions.RequestException as e:
        print(f"Network transmission failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return None


def play_audio(file_path):
    """Plays the returned audio file immediately over system speakers using sounddevice and soundfile."""
    print("Playing Response...")
    if not os.path.exists(file_path):
        print(f"Error: Playback file {file_path} not found.")
        return

    # Add small delays and retries to handle potential OS file locks on Windows
    for attempt in range(5):
        try:
            data, fs = sf.read(file_path, dtype='float32')
            sd.play(data, fs)
            sd.wait()  # Wait until playback is finished
            print("Playback finished.")
            return
        except Exception as e:
            if attempt == 4:
                print(f"Error during audio playback: {e}")
            else:
                time.sleep(0.1)


def main():
    # Setup a thread-safe queue for the recording callback
    q = queue.Queue()

    def recording_callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        q.put(indata.copy())

    # Query default input device and default rate
    try:
        input_info = sd.query_devices(kind='input')
        default_rate = int(input_info['default_samplerate'])
        print(f"Detected default input device: {input_info['name']} (Native Rate: {default_rate} Hz)")
    except Exception as e:
        default_rate = 16000
        print(f"Warning: Could not query default input device ({e}). Defaulting to 16000 Hz.")

    print("\n================ X-Plane 11 AI ATC Live Interface ================")
    if KEYBOARD_AVAILABLE:
        print("Push-to-Talk (PTT) Mode: PRESS and HOLD the SPACEBAR to speak.")
    else:
        print("Fallback Mode: Press ENTER to start, and ENTER to stop recording.")
    print("==================================================================\n")
    
    prompted = False
    
    try:
        while True:
            if KEYBOARD_AVAILABLE:
                if not prompted:
                    print("Ready. Press and hold SPACEBAR to speak...")
                    prompted = True
                
                # Check for spacebar press
                if keyboard.is_pressed('space'):
                    prompted = False
                    print("\nRecording...")
                    
                    recorded_data = []
                    
                    # Clear queue
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            break
                    
                    # Open stream
                    stream = None
                    rates_to_try = [16000, default_rate, 44100, 48000]
                    rates_to_try = list(dict.fromkeys([int(r) for r in rates_to_try if r]))
                    
                    actual_rate = 16000
                    for rate in rates_to_try:
                        try:
                            stream = sd.InputStream(samplerate=rate, channels=CHANNELS, callback=recording_callback, dtype='int16')
                            stream.start()
                            actual_rate = rate
                            break
                        except Exception as e:
                            print(f"  Rate {rate} Hz failed: {e}")
                            continue
                            
                    if stream is not None:
                        print(f"Successfully opened microphone stream at {actual_rate} Hz.")
                        
                        # Record while space is held down
                        while keyboard.is_pressed('space'):
                            while not q.empty():
                                try:
                                    block = q.get_nowait()
                                    recorded_data.append(block)
                                except queue.Empty:
                                    break
                            time.sleep(0.005)
                        
                        # Stop and close the stream
                        stream.stop()
                        stream.close()
                        
                        # Drain remaining queue items
                        while not q.empty():
                            try:
                                block = q.get_nowait()
                                recorded_data.append(block)
                            except queue.Empty:
                                break
                        
                        print("Recording finished. Saving audio...")
                        if recorded_data:
                            audio_np = np.concatenate(recorded_data, axis=0)
                            sf.write(AUDIO_FILE_INPUT, audio_np, actual_rate, subtype='PCM_16')
                            print(f"Saved recording to {AUDIO_FILE_INPUT}")
                        else:
                            print("Warning: No audio data recorded.")
                            continue
                            
                    else:
                        print("\n[Microphone Error] Could not open any audio input stream using sounddevice.")
                        print("  [Tip] Since your Windows microphone driver is currently blocked by OS privacy settings,")
                        print("        you can record your question (e.g. 'What airport am I at?') using Windows Sound Recorder,")
                        print("        save it as 'test.wav' in the parent folder, and this script will automatically transmit it!")
                        
                        # Fallback to test.wav if microphone fails
                        fallback_file = None
                        for path in ["test.wav", "../test.wav", "ATC-AI/test.wav"]:
                            if os.path.exists(path):
                                fallback_file = path
                                break
                                
                        if fallback_file:
                            print(f"\n-> Fallback Mode: Using existing '{fallback_file}' to test the rest of the pipeline...")
                            try:
                                shutil.copy(fallback_file, AUDIO_FILE_INPUT)
                            except Exception as cp_err:
                                print(f"Error copying fallback file: {cp_err}")
                                continue
                        else:
                            print("-> No fallback 'test.wav' found in workspace. Skipping transmission.")
                            time.sleep(1.0)
                            continue
                    
                    # Fetch telemetry
                    telemetry = fetch_xplane_telemetry()
                    
                    # Transmit to AWS AI backend
                    response_file = transmit_audio_and_telemetry(AUDIO_FILE_INPUT, telemetry)
                    
                    # Play back response if downloaded successfully
                    if response_file:
                        play_audio(response_file)
                        
                time.sleep(0.05)  # Avoid high CPU utilization checking the key
                
            else:
                # Fallback Enter-key mode
                input("Press [ENTER] to start recording...")
                print("\nRecording...")
                
                recorded_data = []
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
                        
                stream = None
                rates_to_try = [16000, default_rate, 44100, 48000]
                rates_to_try = list(dict.fromkeys([int(r) for r in rates_to_try if r]))
                
                actual_rate = 16000
                for rate in rates_to_try:
                    try:
                        stream = sd.InputStream(samplerate=rate, channels=CHANNELS, callback=recording_callback, dtype='int16')
                        stream.start()
                        actual_rate = rate
                        break
                    except Exception as e:
                        print(f"  Rate {rate} Hz failed: {e}")
                        continue
                        
                if stream is not None:
                    print(f"Successfully opened microphone stream at {actual_rate} Hz.")
                    
                    # Wait for enter key to stop
                    input("Recording active. Press [ENTER] to stop...")
                    
                    stream.stop()
                    stream.close()
                    
                    while not q.empty():
                        try:
                            block = q.get_nowait()
                            recorded_data.append(block)
                        except queue.Empty:
                            break
                            
                    print("Recording finished. Saving audio...")
                    if recorded_data:
                        audio_np = np.concatenate(recorded_data, axis=0)
                        sf.write(AUDIO_FILE_INPUT, audio_np, actual_rate, subtype='PCM_16')
                        print(f"Saved recording to {AUDIO_FILE_INPUT}")
                    else:
                        print("Warning: No audio data recorded.")
                        continue
                else:
                    print("\n[Microphone Error] Could not open any audio input stream using sounddevice.")
                    print("  [Tip] Since your Windows microphone driver is currently blocked by OS privacy settings,")
                    print("        you can record your question (e.g. 'What airport am I at?') using Windows Sound Recorder,")
                    print("        save it as 'test.wav' in the parent folder, and this script will automatically transmit it!")
                    
                    fallback_file = None
                    for path in ["test.wav", "../test.wav", "ATC-AI/test.wav"]:
                        if os.path.exists(path):
                            fallback_file = path
                            break
                            
                    if fallback_file:
                        print(f"\n-> Fallback Mode: Using existing '{fallback_file}' to test the rest of the pipeline...")
                        try:
                            shutil.copy(fallback_file, AUDIO_FILE_INPUT)
                        except Exception as cp_err:
                            print(f"Error copying fallback file: {cp_err}")
                            continue
                    else:
                        print("-> No fallback 'test.wav' found in workspace. Skipping transmission.")
                        time.sleep(1.0)
                        continue
                
                # Fetch telemetry
                telemetry = fetch_xplane_telemetry()
                
                # Transmit
                response_file = transmit_audio_and_telemetry(AUDIO_FILE_INPUT, telemetry)
                
                # Playback
                if response_file:
                    play_audio(response_file)
                    
    except KeyboardInterrupt:
        print("\nExiting program...")


if __name__ == "__main__":
    main()
