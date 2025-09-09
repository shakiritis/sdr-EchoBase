# EchoBase: Speech-to-Text over the Air using Software-Defined Radios and OFDM

EchoBase is a real-time speech communication system that combines advanced speech recognition with wireless transmission. The project leverages OpenAI’s Faster-Whisper model to convert spoken language into text with high accuracy, and then employs GNU Radio with software-defined radios (SDRs) to modulate and transmit the transcribed text using orthogonal frequency-division multiplexing (OFDM).

## System Implementation

### Overview

The EchoBase system integrates **Faster-Whisper** for real-time speech recognition with **GNU Radio** and **SDRs** for over-the-air text transmission using OFDM. The complete pipeline is illustrated below:

### Steps

1. **Speech-to-Text Setup**  
The host computer connected to the **transmitting SDR (TX)** runs the speech recognition pipeline.  
Required packages include **Faster-Whisper**, **sounddevice**, and **scipy**, along with PortAudio for microphone access.  

Install with:
```bash
pip install faster-whisper sounddevice scipy
sudo apt install portaudio19-dev   # Linux dependency
```

2. **Speech Transcription (`speech2txt.py`)**  
- Continuously records microphone input.  
- Uses **Faster-Whisper** to transcribe speech to text in near real time.  
- Maintains a sliding buffer of recent audio (≈30s) and applies VAD to filter silence.  
- Writes recognized text segments into **`transcript.txt`**, which acts as the message source for the SDR transmission.  

3. **Transmission (`TX.py`)**  
- Reads updated lines from `transcript.txt`.  
- Encodes each line into frames with **header + payload**.  
- **Modulation options:**  
  - Header: **BPSK** (for synchronization and robustness).  
  - Payload: **BPSK** (default) or **QPSK** (works well too).  
- Uses **GNU Radio** with a **USRP N210** SDR to transmit the frames over OFDM.  

4. **Reception (`RX.py`)**  
- Runs on a second host with a receiving **USRP N210** SDR.  
- Synchronizes on the BPSK header.  
- Demodulates the payload (BPSK or QPSK).  
- Reconstructs and displays the received text, effectively streaming transcribed speech from TX to RX over RF.

## Requirements:

* faster-whisper (1.2.0)
* gnuradio-companion (3.8+)

## Usage

* On the TX host: python3 speech2txt.py
* On the TX SDR: python3 TX.py
* On the RX SDR: python3 RX.py
