/**
 * audio-processor.js — EchoPilot AudioWorklet Processor
 *
 * Runs on the dedicated AudioWorkletGlobalScope (audio rendering thread),
 * NOT the main JS thread. This prevents GC pauses, slow DOM repaints, or
 * event handler backpressure from causing dropped audio samples.
 *
 * The processor fires in fixed 128-sample quanta (the Web Audio API's
 * render quantum). We accumulate samples into a larger buffer (matching the
 * old ScriptProcessorNode's 2048-sample frame size) before posting to the
 * main thread so the WebSocket send rate is identical to before.
 *
 * Message sent to main thread: ArrayBuffer containing Int16Array PCM data
 * at 16kHz mono.
 */
class PCMProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        // Accumulate 2048 samples before flushing (matches previous ScriptProcessor frame size)
        // At 16kHz: 2048 samples = 128ms per flush — same latency as before
        this._bufferSize = (options && options.processorOptions && options.processorOptions.bufferSize) || 2048;
        this._buffer = new Float32Array(this._bufferSize);
        this._writeOffset = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || !input[0]) {
            return true; // keep processor alive even if no input connected yet
        }

        const channel = input[0]; // mono channel (we request 1 input channel)

        for (let i = 0; i < channel.length; i++) {
            this._buffer[this._writeOffset++] = channel[i];

            if (this._writeOffset >= this._bufferSize) {
                // Convert float32 [-1, 1] → int16 [-32768, 32767]
                const int16 = new Int16Array(this._bufferSize);
                for (let j = 0; j < this._bufferSize; j++) {
                    const s = Math.max(-1, Math.min(1, this._buffer[j]));
                    int16[j] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                // Transfer ownership of the ArrayBuffer to the main thread (zero-copy)
                this.port.postMessage(int16.buffer, [int16.buffer]);
                this._writeOffset = 0;
            }
        }

        return true; // returning true keeps the node alive
    }
}

registerProcessor('pcm-processor', PCMProcessor);
