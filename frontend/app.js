// Core UI Elements
const callBtn = document.getElementById('call-btn');
const endCallBtn = document.getElementById('end-call-btn');
const callActionArea = document.querySelector('.call-action-area');
const callInterface = document.getElementById('call-interface');
const statusText = document.getElementById('status-text');
const transcriptArea = document.getElementById('transcript-area');
let thinkingIndicator = null;

// WebSocket and Audio state
let ws = null;
let processor = null;
let audioInput = null;
let audioContext = null;       // 16kHz context for RECORDING (mic -> backend)
let playbackContext = null;    // Default sample rate context for PLAYBACK (TTS -> speakers)
let isConnected = false;
let assistantSpeaking = false;
let mediaStream = null;        // Store reference to close tracks later

// Call Timer state
let callTimerInterval = null;
let callStartTime = 0;
const callTimerEl = document.getElementById('call-timer');
const visualizerEl = document.querySelector('.visualizer');

const STATUS_MAP = {
    "listening": "🎤 Listening...",
    "transcribing": "⚡ Processing...",
    "thinking": "💭 Thinking...",
    "speaking": "🔊 Speaking... (interrupt me anytime!)"
};

// Audio Configuration (standard 16kHz for STT)
const AUDIO_SAMPLE_RATE = 16000;

callBtn.addEventListener('click', startCall);
endCallBtn.addEventListener('click', endCall);

async function startCall() {
    try {
        statusText.textContent = "Requesting microphone access...";
        
        // 1. Get Microphone Access (with echo cancellation for barge-in)
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });
        
        statusText.textContent = "Connecting to AI Assistant...";
        
        // 2. Setup WebSocket Connection
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        let wsHost = window.location.host || '127.0.0.1:8000';
        if (wsHost.includes(':3000')) {
            wsHost = wsHost.replace(':3000', ':8000');
        }
        ws = new WebSocket(`${wsProtocol}//${wsHost}/ws/audio`);
        
        ws.onopen = () => {
            isConnected = true;
            console.log("WebSocket connected");
            
            // UI Transition
            callActionArea.classList.add('hidden');
            callInterface.classList.remove('hidden');
            setTimeout(() => callInterface.classList.add('active'), 50);
            
            transcriptArea.innerHTML = ""; // Clear on start
            
            // Start Timer
            callStartTime = Date.now();
            callTimerEl.textContent = "00:00";
            callTimerInterval = setInterval(() => {
                const elapsed = Math.floor((Date.now() - callStartTime) / 1000);
                const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
                const secs = String(elapsed % 60).padStart(2, '0');
                callTimerEl.textContent = `${mins}:${secs}`;
            }, 1000);
            
            // 3. Start Recording & Streaming (creates audioContext at 16kHz)
            startStreaming(stream);
            
            // 4. Create a SEPARATE playback context at the browser's default sample rate
            playbackContext = new (window.AudioContext || window.webkitAudioContext)();
        };
        
        ws.onmessage = (event) => {
            if (typeof event.data === "string") {
                const data = JSON.parse(event.data);
                
                if (data.type === "status") {
                    statusText.textContent = STATUS_MAP[data.message] || data.message;
                    assistantSpeaking = (data.message === "speaking");
                    
                    if (assistantSpeaking) {
                        visualizerEl.classList.add('speaking');
                        callInterface.classList.add('speaking-state');
                    } else {
                        visualizerEl.classList.remove('speaking');
                        callInterface.classList.remove('speaking-state');
                    }
                    
                    if (data.message === "thinking") {
                        showThinkingIndicator();
                    } else {
                        removeThinkingIndicator();
                    }
                }
                
                if (data.type === "interrupt") {
                    if (currentSource) {
                        try {
                            currentSource.onended = null;
                            currentSource.stop();
                        } catch (e) { /* ignore */ }
                        currentSource = null;
                    }
                    assistantSpeaking = false;
                    visualizerEl.classList.remove('speaking');
                    callInterface.classList.remove('speaking-state');
                    statusText.textContent = "🎤 Listening...";
                    // Tell backend we've stopped playback
                    if (isConnected && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({type: "playback_ended"}));
                    }
                }
                
                if (data.type === "transcript" && data.final) {
                    const speakerLabel = data.speaker === "assistant" ? "Lumina Assistant" : "You";
                    appendTranscriptLine(speakerLabel, data.text);
                }
            } else {
                // Binary data received (TTS audio)
                playReceivedAudio(event.data);
            }
        };
        
        ws.onclose = () => {
            console.log("WebSocket disconnected");
            endCall();
        };
        
        ws.onerror = (error) => {
            console.error("WebSocket Error:", error);
            statusText.textContent = "Connection error. Make sure the backend is running.";
            endCall();
        };

    } catch (err) {
        console.error("Error starting call:", err);
        statusText.textContent = "Microphone access denied or not available.";
    }
}

function startStreaming(stream) {
    mediaStream = stream; // Save reference for cleanup
    
    // This context runs at 16kHz for clean STT recording
    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: AUDIO_SAMPLE_RATE });
    audioInput = audioContext.createMediaStreamSource(stream);
    processor = audioContext.createScriptProcessor(4096, 1, 1);
    
    audioInput.connect(processor);
    processor.connect(audioContext.destination);
    
    processor.onaudioprocess = (e) => {
        // Continuous streaming: keep sending even if assistantSpeaking is true!
        if (!isConnected || ws.readyState !== WebSocket.OPEN) return;
        
        const float32Array = e.inputBuffer.getChannelData(0);
        const int16Array = new Int16Array(float32Array.length);
        for (let i = 0; i < float32Array.length; i++) {
            let s = Math.max(-1, Math.min(1, float32Array[i]));
            int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        ws.send(int16Array.buffer);
    };
}

let currentSource = null;  // Track currently playing audio to prevent overlap

async function playReceivedAudio(blob) {
    // Use the dedicated playback context (NOT the 16kHz recording context)
    if (!playbackContext) return;
    
    try {
        // Stop any currently playing audio to prevent overlapping voices
        if (currentSource) {
            try {
                currentSource.onended = null;  // Remove callback before stopping
                currentSource.stop();
            } catch (e) { /* already stopped */ }
        }

        const arrayBuffer = await blob.arrayBuffer();
        const audioBuffer = await playbackContext.decodeAudioData(arrayBuffer);
        
        const source = playbackContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(playbackContext.destination);
        source.onended = () => {
            // Signal that speaking is done so mic can resume
            currentSource = null;
            assistantSpeaking = false;
            visualizerEl.classList.remove('speaking');
            callInterface.classList.remove('speaking-state');
            if (isConnected && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({type: "playback_ended"}));
            }
        };
        currentSource = source;
        source.start(0);
    } catch (err) {
        console.error("Error playing TTS audio:", err);
        currentSource = null;
        assistantSpeaking = false;
        visualizerEl.classList.remove('speaking');
        callInterface.classList.remove('speaking-state');
    }
}

function endCall() {
    isConnected = false;
    assistantSpeaking = false;
    visualizerEl.classList.remove('speaking');
    callInterface.classList.remove('speaking-state');
    
    if (callTimerInterval) {
        clearInterval(callTimerInterval);
        callTimerInterval = null;
    }
    
    if (processor) {
        processor.disconnect();
        processor = null;
    }
    if (audioInput) {
        audioInput.disconnect();
        audioInput = null;
    }
    
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    
    if (ws) {
        ws.close();
        ws = null;
    }
    
    // Close BOTH audio contexts
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    if (playbackContext) {
        playbackContext.close();
        playbackContext = null;
    }
    
    // UI Reset
    callInterface.classList.remove('active');
    setTimeout(() => {
        callInterface.classList.add('hidden');
        callActionArea.classList.remove('hidden');
        statusText.textContent = "Call ended. Ready to connect.";
        transcriptArea.innerHTML = "";
    }, 400);
}

function appendTranscriptLine(speaker, text) {
    removeThinkingIndicator(); // Just in case
    const isAssistant = speaker === "assistant";
    
    const bubble = document.createElement("div");
    bubble.className = `chat-bubble ${isAssistant ? 'bubble-assistant' : 'bubble-user'}`;
    bubble.textContent = text;
    
    transcriptArea.appendChild(bubble);
    transcriptArea.scrollTop = transcriptArea.scrollHeight;
}

function showThinkingIndicator() {
    if (thinkingIndicator) return;
    thinkingIndicator = document.createElement("div");
    thinkingIndicator.className = "chat-bubble bubble-thinking";
    thinkingIndicator.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    `;
    transcriptArea.appendChild(thinkingIndicator);
    transcriptArea.scrollTop = transcriptArea.scrollHeight;
}

function removeThinkingIndicator() {
    if (thinkingIndicator && thinkingIndicator.parentNode) {
        thinkingIndicator.parentNode.removeChild(thinkingIndicator);
    }
    thinkingIndicator = null;
}
