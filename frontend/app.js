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
            const heroSection = document.querySelector('.hero-section');
            if (heroSection) heroSection.classList.add('in-call');
            
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
    const heroSection = document.querySelector('.hero-section');
    if (heroSection) heroSection.classList.remove('in-call');

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

// ==========================================
// DATABASE MODAL & DATA MANAGEMENT
// ==========================================

const openDbBtn = document.getElementById('open-db-btn');
const closeDbBtn = document.getElementById('close-db-btn');
const reseedDbBtn = document.getElementById('reseed-db-btn');
const dbModal = document.getElementById('db-modal');
const tabButtons = document.querySelectorAll('.tab-btn');
const tabPanes = document.querySelectorAll('.tab-pane');

// Open / Close Modal
if (openDbBtn) {
    openDbBtn.addEventListener('click', () => {
        dbModal.classList.remove('hidden');
        loadDatabaseData();
    });
}

if (closeDbBtn) {
    closeDbBtn.addEventListener('click', () => {
        dbModal.classList.add('hidden');
    });
}

// Close on backdrop click
if (dbModal) {
    dbModal.addEventListener('click', (e) => {
        if (e.target === dbModal) {
            dbModal.classList.add('hidden');
        }
    });
}

// Tab Switching
tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const targetTab = btn.getAttribute('data-tab');
        
        tabButtons.forEach(b => b.classList.remove('active'));
        tabPanes.forEach(pane => pane.classList.remove('active'));
        
        btn.classList.add('active');
        const activePane = document.getElementById(targetTab);
        if (activePane) activePane.classList.add('active');
    });
});

// Reseed Sample Data Button
if (reseedDbBtn) {
    reseedDbBtn.addEventListener('click', async () => {
        if (!confirm("Are you sure you want to reset and re-seed the sample healthcare records?")) return;
        
        reseedDbBtn.disabled = true;
        reseedDbBtn.innerHTML = `<i class="ph-bold ph-spinner"></i> Resetting...`;
        
        try {
            const res = await fetch('/api/database/seed', { method: 'POST' });
            const data = await res.json();
            alert("Database re-seeded successfully with clean sample records!");
            loadDatabaseData();
        } catch (e) {
            console.error("Error reseeding:", e);
            alert("Failed to reset database. Backend might be unreachable.");
        } finally {
            reseedDbBtn.disabled = false;
            reseedDbBtn.innerHTML = `<i class="ph-bold ph-arrow-clockwise"></i> Reset Sample Data`;
        }
    });
}

// Load all DB Tables & Stats
async function loadDatabaseData() {
    try {
        // 1. Overview & Stats
        fetch('/api/database/overview')
            .then(res => res.json())
            .then(data => {
                if (data.counts) {
                    document.getElementById('stat-patients').textContent = data.counts.patients ?? 0;
                    document.getElementById('stat-appointments').textContent = data.counts.appointments ?? 0;
                    document.getElementById('stat-doctors').textContent = data.counts.doctors ?? 0;
                    document.getElementById('stat-calls').textContent = data.counts.call_logs ?? 0;
                }
            })
            .catch(console.error);

        // 2. Patients & Sensitive Records
        fetch('/api/database/patients')
            .then(res => res.json())
            .then(data => renderPatientsTable(data.patients || []))
            .catch(console.error);

        // 3. Appointments
        fetch('/api/database/appointments')
            .then(res => res.json())
            .then(data => renderAppointmentsTable(data.appointments || []))
            .catch(console.error);

        // 4. Doctors
        fetch('/api/database/doctors')
            .then(res => res.json())
            .then(data => renderDoctorsTable(data.doctors || []))
            .catch(console.error);

        // 5. Hospital Info
        fetch('/api/database/hospitals')
            .then(res => res.json())
            .then(data => renderHospitalInfo(data.hospitals || []))
            .catch(console.error);

        // 6. Call Logs
        fetch('/api/database/call-logs')
            .then(res => res.json())
            .then(data => renderCallLogsTable(data.call_logs || []))
            .catch(console.error);

    } catch (err) {
        console.error("Failed to load database view:", err);
    }
}

function renderPatientsTable(patients) {
    const tbody = document.getElementById('patients-tbody');
    if (!tbody) return;
    
    if (patients.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" class="text-center">No patient records found in database.</td></tr>`;
        return;
    }

    tbody.innerHTML = patients.map(p => `
        <tr>
            <td><span class="badge-mrn">${escapeHtml(p.medical_record_number || 'N/A')}</span></td>
            <td>
                <strong>${escapeHtml(p.name)}</strong>
                <div style="font-size: 0.75rem; color: #94a3b8;">DOB: ${escapeHtml(p.date_of_birth || 'N/A')} (SSN: ***-**-${escapeHtml(p.national_id_ssn_last4 || '****')})</div>
            </td>
            <td>
                <div>${escapeHtml(p.phone)}</div>
                <div style="font-size: 0.75rem; color: #94a3b8;">${escapeHtml(p.email || '')}</div>
            </td>
            <td>
                <div><span class="badge-sensitive">Blood: ${escapeHtml(p.blood_group || 'Unknown')}</span></div>
                <div style="font-size: 0.8rem; color: #f87171; margin-top: 4px;">⚠️ ${escapeHtml(p.allergies || 'NKDA')}</div>
            </td>
            <td style="font-size: 0.85rem; color: #cbd5e1;">${escapeHtml(p.chronic_conditions || 'None')}</td>
            <td style="font-size: 0.82rem; font-family: monospace; color: #38bdf8;">${escapeHtml(p.current_medications || 'None')}</td>
            <td>
                <div style="font-weight: 500;">${escapeHtml(p.insurance_provider || 'Self Pay')}</div>
                <div style="font-size: 0.75rem; color: #94a3b8; font-family: monospace;">Pol: ${escapeHtml(p.insurance_policy_number || 'N/A')}</div>
            </td>
            <td style="font-size: 0.8rem; max-width: 220px; color: #94a3b8;">${escapeHtml(p.confidential_notes || '')}</td>
        </tr>
    `).join('');
}

function renderAppointmentsTable(appointments) {
    const tbody = document.getElementById('appointments-tbody');
    if (!tbody) return;

    if (appointments.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" class="text-center">No appointments found. Book one via voice agent!</td></tr>`;
        return;
    }

    tbody.innerHTML = appointments.map(a => `
        <tr>
            <td><span class="badge-code">${escapeHtml(a.booking_code)}</span></td>
            <td>
                <strong>${escapeHtml(a.patient_name)}</strong>
                <div style="font-size: 0.75rem; color: #94a3b8;">${escapeHtml(a.patient_phone || '')}</div>
            </td>
            <td>
                <div style="font-weight: 600; color: #38bdf8;">${escapeHtml(a.doctor_name || 'Assigned Physician')}</div>
                <div style="font-size: 0.75rem; color: #94a3b8;">${escapeHtml(a.doctor_department || 'Clinic')}</div>
            </td>
            <td><strong>${escapeHtml(a.service_name)}</strong></td>
            <td>
                <div>📅 ${escapeHtml(a.appointment_date)}</div>
                <div style="font-size: 0.8rem; color: #2dd4bf;">⏰ ${escapeHtml(a.appointment_time)}</div>
            </td>
            <td><span class="badge-status-confirmed">${escapeHtml(a.status)}</span></td>
            <td style="font-size: 0.8rem; color: #94a3b8;">${escapeHtml(a.booked_via || 'AI_VOICE_AGENT')}</td>
            <td style="font-size: 0.8rem; color: #cbd5e1;">${escapeHtml(a.patient_notes || '')}</td>
        </tr>
    `).join('');
}

function renderDoctorsTable(doctors) {
    const tbody = document.getElementById('doctors-tbody');
    if (!tbody) return;

    tbody.innerHTML = doctors.map(d => `
        <tr>
            <td>#${d.id}</td>
            <td><strong>${escapeHtml(d.name)}</strong></td>
            <td><span style="color: #38bdf8; font-weight: 500;">${escapeHtml(d.department)}</span></td>
            <td>${escapeHtml(d.specialization)}</td>
            <td><span class="badge-sensitive" style="font-family: monospace;">${escapeHtml(d.license_number)}</span></td>
            <td><strong style="color: #34d399;">$${parseFloat(d.consultation_fee).toFixed(2)}</strong></td>
            <td style="font-size: 0.8rem;">
                <div>${escapeHtml(d.available_days)}</div>
                <div style="color: #94a3b8;">${escapeHtml(d.available_hours)}</div>
            </td>
            <td style="font-size: 0.8rem; color: #94a3b8;">${escapeHtml(d.contact_email)}</td>
        </tr>
    `).join('');
}

function renderHospitalInfo(hospitals) {
    const container = document.getElementById('hospital-details-cards');
    if (!container) return;

    if (hospitals.length === 0) {
        container.innerHTML = `<p>No hospital records found.</p>`;
        return;
    }

    const h = hospitals[0];
    container.innerHTML = `
        <div class="info-card">
            <div class="info-card-header">
                <i class="ph-fill ph-hospital"></i>
                <h3>Facility Profile</h3>
            </div>
            <div class="info-row">
                <span class="info-label">Hospital Name:</span>
                <span class="info-val">${escapeHtml(h.name)}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Branch / Plaza:</span>
                <span class="info-val">${escapeHtml(h.branch || 'Main')}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Address:</span>
                <span class="info-val">${escapeHtml(h.address)}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Direct Phone:</span>
                <span class="info-val">${escapeHtml(h.direct_phone)}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Secure Admin Email:</span>
                <span class="info-val">${escapeHtml(h.email)}</span>
            </div>
        </div>

        <div class="info-card" style="border-color: rgba(239, 68, 68, 0.3);">
            <div class="info-card-header">
                <i class="ph-fill ph-shield-check" style="color: #f87171;"></i>
                <h3>Sensitive Legal & Compliance Records</h3>
            </div>
            <div class="info-row">
                <span class="info-label">State Medical Reg #:</span>
                <span class="info-val sensitive">${escapeHtml(h.registration_number)}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Tax ID / EIN:</span>
                <span class="info-val sensitive">${escapeHtml(h.tax_id_ein)}</span>
            </div>
            <div class="info-row">
                <span class="info-label">DEA Schedule License:</span>
                <span class="info-val sensitive">${escapeHtml(h.dea_license_number)}</span>
            </div>
            <div class="info-row">
                <span class="info-label">HIPAA Audit Certificate:</span>
                <span class="info-val sensitive">${escapeHtml(h.hipaa_compliance_id)}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Emergency ER Hotline:</span>
                <span class="info-val" style="color: #f97316; font-weight: 700;">${escapeHtml(h.emergency_hotline)}</span>
            </div>
        </div>
    `;
}

function renderCallLogsTable(logs) {
    const tbody = document.getElementById('call-logs-tbody');
    if (!tbody) return;

    if (logs.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" class="text-center">No call logs recorded yet. Place a call to see live session transcripts!</td></tr>`;
        return;
    }

    tbody.innerHTML = logs.map(l => `
        <tr>
            <td><span class="badge-code" style="font-size: 0.75rem;">${escapeHtml(l.session_id)}</span></td>
            <td>
                <strong>${escapeHtml(l.caller_name || 'Guest')}</strong>
                <div style="font-size: 0.75rem; color: #94a3b8;">${escapeHtml(l.caller_phone || 'Unknown')}</div>
            </td>
            <td>${escapeHtml(l.service_requested || 'General')}</td>
            <td><strong style="color: #38bdf8;">${l.call_duration_seconds}s</strong></td>
            <td><span class="badge-status-confirmed">${escapeHtml(l.status)}</span></td>
            <td style="font-size: 0.75rem; color: #94a3b8;">${escapeHtml(l.created_at || '')}</td>
            <td style="font-size: 0.78rem; font-family: monospace; white-space: pre-wrap; max-width: 320px; color: #cbd5e1; background: rgba(0,0,0,0.2); padding: 6px; border-radius: 6px;">${escapeHtml(l.transcript || 'No transcript text')}</td>
        </tr>
    `).join('');
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// ==========================================
// MODE SWITCHER (VOICE CALL VS CHATBOT)
// ==========================================

const modeVoiceBtn = document.getElementById('mode-voice-btn');
const modeChatBtn = document.getElementById('mode-chat-btn');
const voiceModeSection = document.getElementById('voice-mode-section');
const chatModeSection = document.getElementById('chat-mode-section');

if (modeVoiceBtn && modeChatBtn) {
    modeVoiceBtn.addEventListener('click', () => {
        modeVoiceBtn.classList.add('active');
        modeChatBtn.classList.remove('active');
        voiceModeSection.classList.remove('hidden');
        chatModeSection.classList.add('hidden');
    });

    modeChatBtn.addEventListener('click', () => {
        modeChatBtn.classList.add('active');
        modeVoiceBtn.classList.remove('active');
        chatModeSection.classList.remove('hidden');
        voiceModeSection.classList.add('hidden');
        
        // If voice call is active, end it
        if (isConnected) {
            endCall();
        }
        
        // Initialize chat if empty
        if (chatMessagesArea && chatMessagesArea.children.length === 0) {
            initChatbot();
        }
    });
}

// ==========================================
// INTERACTIVE TEXT CHATBOT LOGIC
// ==========================================

const chatMessagesArea = document.getElementById('chat-messages-area');
const chatSuggestionsArea = document.getElementById('chat-suggestions-area');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const resetChatBtn = document.getElementById('reset-chat-btn');

let chatSessionId = `chat_sess_${Date.now()}`;
let chatThinkingEl = null;

// Initialize Chatbot with Greeting
async function initChatbot() {
    chatMessagesArea.innerHTML = "";
    chatSuggestionsArea.innerHTML = "";
    
    appendChatBubble('assistant', "Hey there! I'm Lumina from the health clinic. What kind of appointment or health service can I help you book today?");
    renderChatSuggestions([
        "General Checkup",
        "Cardiology Consultation",
        "Dermatology Skin Exam",
        "Pediatric Screening",
        "Orthopedic Evaluation"
    ]);
}

// Handle Message Submit
if (chatForm) {
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = chatInput.value.trim();
        if (!text) return;
        
        // Clear input field
        chatInput.value = "";
        
        // Append user bubble
        appendChatBubble('user', text);
        chatSuggestionsArea.innerHTML = ""; // Clear suggestions while thinking
        
        // Show thinking indicator
        showChatThinking();
        
        try {
            const res = await fetch('/api/chat/message', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: chatSessionId,
                    message: text
                })
            });
            
            const data = await res.json();
            removeChatThinking();
            
            if (data.reply) {
                appendChatBubble('assistant', data.reply);
            }
            
            // Render smart suggestion chips
            if (data.quick_replies && data.quick_replies.length > 0) {
                renderChatSuggestions(data.quick_replies);
            }
            
            // If appointment was booked, notify user and refresh database summary
            if (data.is_booked) {
                fetch('/api/database/overview')
                    .then(r => r.json())
                    .then(ov => {
                        if (ov.counts) {
                            document.getElementById('stat-patients').textContent = ov.counts.patients ?? 0;
                            document.getElementById('stat-appointments').textContent = ov.counts.appointments ?? 0;
                        }
                    })
                    .catch(console.error);
            }
            
        } catch (err) {
            console.error("Chat error:", err);
            removeChatThinking();
            appendChatBubble('assistant', "Sorry, I ran into a connection error. Please make sure the backend server is running.");
        }
    });
}

// Handle Reset Chat Button
if (resetChatBtn) {
    resetChatBtn.addEventListener('click', async () => {
        resetChatBtn.disabled = true;
        chatSessionId = `chat_sess_${Date.now()}`;
        
        try {
            await fetch('/api/chat/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: chatSessionId })
            });
        } catch (e) {
            console.error("Reset error:", e);
        }
        
        initChatbot();
        resetChatBtn.disabled = false;
    });
}

function appendChatBubble(speaker, text) {
    if (!chatMessagesArea) return;
    
    const isAssistant = speaker === 'assistant';
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${isAssistant ? 'bubble-assistant' : 'bubble-user'}`;
    bubble.textContent = text;
    
    chatMessagesArea.appendChild(bubble);
    chatMessagesArea.scrollTop = chatMessagesArea.scrollHeight;
}

function renderChatSuggestions(suggestions) {
    if (!chatSuggestionsArea) return;
    chatSuggestionsArea.innerHTML = "";
    
    suggestions.forEach(text => {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'suggestion-chip';
        chip.textContent = text;
        chip.addEventListener('click', () => {
            chatInput.value = text;
            chatForm.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
        });
        chatSuggestionsArea.appendChild(chip);
    });
}

function showChatThinking() {
    if (chatThinkingEl || !chatMessagesArea) return;
    chatThinkingEl = document.createElement('div');
    chatThinkingEl.className = "chat-bubble bubble-thinking";
    chatThinkingEl.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    `;
    chatMessagesArea.appendChild(chatThinkingEl);
    chatMessagesArea.scrollTop = chatMessagesArea.scrollHeight;
}

function removeChatThinking() {
    if (chatThinkingEl && chatThinkingEl.parentNode) {
        chatThinkingEl.parentNode.removeChild(chatThinkingEl);
    }
    chatThinkingEl = null;
}


