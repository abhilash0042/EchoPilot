/**
 * 3D Interactive AI Avatar Engine for Elena Vance (EchoPilot)
 * Built with Three.js & @pixiv/three-vrm
 * Features: Real-time Audio LipSync, AutoBlink, Natural Breathing, LookAt Gaze,
 * Active Listening micro-nods, and Emotional State morphing.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';

export class AvatarController {
    constructor(canvasElement, options = {}) {
        this.canvas = canvasElement;
        this.container = canvasElement.parentElement;
        this.options = {
            modelUrl: options.modelUrl || 'assets/models/avatar.vrm',
            onLoaded: options.onLoaded || (() => {}),
            onError: options.onError || ((err) => console.error(err)),
            ...options
        };

        this.vrm = null;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.clock = new THREE.Clock();

        // Animation & Aliveness State
        this.state = 'idle'; // 'idle' | 'listening' | 'thinking' | 'speaking'
        this.targetGaze = new THREE.Vector3(0, 1.40, 1.0);
        this.currentGaze = new THREE.Vector3(0, 1.40, 1.0);
        this.pointer = { x: 0, y: 0, active: false };

        // AutoBlink
        this.blinkRemaining = 3.0;
        this.isBlinking = false;
        this.blinkProgress = 0;

        // Micro-Nod & Active Listening
        this.nodTime = 0;
        this.isNodding = false;
        this.headTiltTarget = 0;
        this.currentHeadTilt = 0;

        // Lip-Sync Audio Analyser
        this.audioAnalyser = null;
        this.timeDomainData = null;
        this.frequencyData = null;
        this.currentMouthOpen = 0;
        this.currentVowelI = 0;
        this.currentVowelO = 0;
        this.analyserAttachedAudio = null;

        // Fallback or WebGL support check
        if (!this.checkWebGL()) {
            console.warn("[AvatarController] WebGL not supported on this browser.");
            return;
        }

        this.initThree();
        this.setupEventListeners();
        this.loadModel(this.options.modelUrl);
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);
    }

    checkWebGL() {
        try {
            const canvas = document.createElement('canvas');
            return !!(window.WebGLRenderingContext && 
                (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
        } catch (e) {
            return false;
        }
    }

    initThree() {
        const width = this.canvas.clientWidth || 320;
        const height = this.canvas.clientHeight || 320;

        // 1. Scene
        this.scene = new THREE.Scene();

        // 2. Camera (Focused on face and upper chest)
        this.camera = new THREE.PerspectiveCamera(28, width / height, 0.1, 20.0);
        this.camera.position.set(0, 1.38, 0.92);
        this.camera.lookAt(0, 1.34, 0);

        // 3. Renderer with high-end color grading
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            alpha: true,
            antialias: true,
            powerPreference: 'high-performance'
        });
        this.renderer.setSize(width, height, false);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.15;

        // 4. Studio Lighting Rig
        // Soft ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 1.4);
        this.scene.add(ambientLight);

        // Warm Key Light (top right)
        const keyLight = new THREE.DirectionalLight(0xfff6ec, 2.0);
        keyLight.position.set(1.2, 2.2, 1.5);
        this.scene.add(keyLight);

        // Soft Fill Light (cool left)
        const fillLight = new THREE.DirectionalLight(0xebf4ff, 1.1);
        fillLight.position.set(-1.2, 1.6, 1.0);
        this.scene.add(fillLight);

        // Subtle Rim / Backlight for hair glow
        const rimLight = new THREE.DirectionalLight(0xaad5ff, 1.2);
        rimLight.position.set(0, 2.0, -1.2);
        this.scene.add(rimLight);

        // LookAt target — must be an Object3D so three-vrm can call .getWorldPosition()
        this.lookAtTarget = new THREE.Object3D();
        this.lookAtTarget.position.set(0, 1.38, 0.92);
        this.scene.add(this.lookAtTarget);

        // Resize Observer
        this.resizeObserver = new ResizeObserver(() => this.handleResize());
        if (this.container) {
            this.resizeObserver.observe(this.container);
        }
    }

    handleResize() {
        if (!this.renderer || !this.camera || !this.canvas) return;
        const width = this.canvas.clientWidth || 320;
        const height = this.canvas.clientHeight || 320;
        if (width === 0 || height === 0) return;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height, false);
    }

    setupEventListeners() {
        // Track mouse/pointer movement across screen for natural eye contact
        window.addEventListener('mousemove', (e) => {
            const x = (e.clientX / window.innerWidth) * 2 - 1;
            const y = -(e.clientY / window.innerHeight) * 2 + 1;
            this.pointer.x = x;
            this.pointer.y = y;
            this.pointer.active = true;
        });

        window.addEventListener('mouseleave', () => {
            this.pointer.active = false;
        });
    }

    loadModel(url) {
        const loader = new GLTFLoader();
        loader.register((parser) => new VRMLoaderPlugin(parser));

        console.log(`[AvatarController] Loading 3D VRM model from: ${url}`);
        loader.load(
            url,
            (gltf) => {
                const vrm = gltf.userData.vrm;
                if (!vrm) {
                    console.error("[AvatarController] GLTF model is not a valid VRM!");
                    return;
                }

                this.vrm = vrm;

                // Optimization helpers
                VRMUtils.removeUnnecessaryVertices(gltf.scene);
                VRMUtils.removeUnnecessaryJoints(gltf.scene);
                VRMUtils.rotateVRM0(vrm);

                this.scene.add(vrm.scene);

                // Wire the LookAt target (must be Object3D, not Vector3)
                if (vrm.lookAt && this.lookAtTarget) {
                    vrm.lookAt.target = this.lookAtTarget;
                }

                // Initial cute resting expression: subtle warm smile
                if (vrm.expressionManager) {
                    vrm.expressionManager.setValue('happy', 0.22);
                    vrm.expressionManager.update();
                }

                console.log("[AvatarController] VRM 3D Avatar initialized successfully!");
                if (this.canvas) this.canvas.classList.add('vrm-loaded');
                this.options.onLoaded(vrm);
            },
            (progress) => {
                if (progress.total > 0) {
                    const pct = Math.round((progress.loaded / progress.total) * 100);
                    // Emit progress if desired
                }
            },
            (error) => {
                console.error("[AvatarController] Failed to load VRM model:", error);
                this.options.onError(error);
            }
        );
    }

    /**
     * Attaches Web Audio analyser to an HTMLAudioElement or AudioContext
     */
    attachAudioSource(audioElement) {
        if (!audioElement || this.analyserAttachedAudio === audioElement) return;

        try {
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            if (!AudioContextClass) return;

            if (!this.audioContext) {
                this.audioContext = new AudioContextClass();
            }

            if (this.audioContext.state === 'suspended') {
                this.audioContext.resume();
            }

            if (!this.audioAnalyser) {
                this.audioAnalyser = this.audioContext.createAnalyser();
                this.audioAnalyser.fftSize = 512;
                this.audioAnalyser.smoothingTimeConstant = 0.25;
                this.timeDomainData = new Float32Array(this.audioAnalyser.fftSize);
                this.frequencyData = new Uint8Array(this.audioAnalyser.frequencyBinCount);
            }

            // Create media element source
            if (!audioElement._vrmSourceNode) {
                const sourceNode = this.audioContext.createMediaElementSource(audioElement);
                sourceNode.connect(this.audioAnalyser);
                this.audioAnalyser.connect(this.audioContext.destination);
                audioElement._vrmSourceNode = sourceNode;
            }

            this.analyserAttachedAudio = audioElement;
        } catch (err) {
            console.warn("[AvatarController] Audio analyser attachment note:", err.message);
        }
    }

    /**
     * Updates lip-sync visemes based on real-time audio FFT/RMS
     */
    updateLipSync(delta) {
        if (!this.vrm || !this.vrm.expressionManager) return;
        const expressions = this.vrm.expressionManager;

        if (this.state !== 'speaking' || !this.audioAnalyser) {
            // Smoothly close mouth
            this.currentMouthOpen = THREE.MathUtils.lerp(this.currentMouthOpen, 0, Math.min(delta * 18, 1.0));
            this.currentVowelI = THREE.MathUtils.lerp(this.currentVowelI, 0, Math.min(delta * 18, 1.0));
            this.currentVowelO = THREE.MathUtils.lerp(this.currentVowelO, 0, Math.min(delta * 18, 1.0));

            expressions.setValue('aa', this.currentMouthOpen);
            expressions.setValue('ih', this.currentVowelI);
            expressions.setValue('oh', this.currentVowelO);
            return;
        }

        // Analyze time-domain energy (volume)
        this.audioAnalyser.getFloatTimeDomainData(this.timeDomainData);
        let sum = 0;
        for (let i = 0; i < this.timeDomainData.length; i++) {
            sum += this.timeDomainData[i] * this.timeDomainData[i];
        }
        const rms = Math.sqrt(sum / this.timeDomainData.length);

        // Analyze frequency bins for vowel distinction (Formants)
        this.audioAnalyser.getByteFrequencyData(this.frequencyData);
        // Low bins (100Hz - 600Hz) ~ 'aa' and 'oh'
        const lowEnergy = (this.frequencyData[1] + this.frequencyData[2] + this.frequencyData[3]) / (3 * 255);
        // High bins (1.5kHz - 3.5kHz) ~ 'ee' / 'ih'
        const highEnergy = (this.frequencyData[12] + this.frequencyData[15] + this.frequencyData[18]) / (3 * 255);

        // Calculate targets
        const targetOpen = Math.min(Math.max((rms - 0.015) * 4.2, 0), 1.0);
        const targetI = Math.min(highEnergy * 1.4, 0.7);
        const targetO = Math.min(lowEnergy * targetOpen * 1.2, 0.6);

        // Smooth attack / decay to avoid flutter
        const attackRate = targetOpen > this.currentMouthOpen ? 28 : 14;
        this.currentMouthOpen = THREE.MathUtils.lerp(this.currentMouthOpen, targetOpen, Math.min(delta * attackRate, 1.0));
        this.currentVowelI = THREE.MathUtils.lerp(this.currentVowelI, targetI, Math.min(delta * 14, 1.0));
        this.currentVowelO = THREE.MathUtils.lerp(this.currentVowelO, targetO, Math.min(delta * 14, 1.0));

        expressions.setValue('aa', this.currentMouthOpen);
        expressions.setValue('ih', this.currentVowelI);
        expressions.setValue('oh', this.currentVowelO);
    }

    /**
     * Natural human blinking with randomized intervals
     */
    updateAutoBlink(delta) {
        if (!this.vrm || !this.vrm.expressionManager) return;
        const expressions = this.vrm.expressionManager;

        if (!this.isBlinking) {
            this.blinkRemaining -= delta;
            if (this.blinkRemaining <= 0) {
                this.isBlinking = true;
                this.blinkProgress = 0;
                // Next blink in 2.5 to 5.5 seconds
                this.blinkRemaining = 2.5 + Math.random() * 3.0;
            }
        } else {
            this.blinkProgress += delta * 12.0; // Quick blink (~100ms)
            const blinkWeight = Math.sin(Math.PI * Math.min(this.blinkProgress, 1.0));
            expressions.setValue('blink', blinkWeight);

            if (this.blinkProgress >= 1.0) {
                this.isBlinking = false;
                expressions.setValue('blink', 0);
            }
        }
    }

    /**
     * Head tracking & eye gaze towards user cursor
     */
    updateLookAt(delta) {
        if (!this.vrm || !this.vrm.lookAt || !this.lookAtTarget) return;

        if (this.state === 'thinking') {
            // Contemplative gaze slightly upward-right
            this.targetGaze.set(0.18, 1.52, 0.85);
        } else if (this.pointer.active) {
            // Smoothly follow user cursor within natural human gaze limits
            const gazeX = THREE.MathUtils.clamp(this.pointer.x * 0.45, -0.4, 0.4);
            const gazeY = 1.38 + THREE.MathUtils.clamp(this.pointer.y * 0.25, -0.2, 0.25);
            this.targetGaze.set(gazeX, gazeY, 0.9);
        } else {
            // Look directly at user / camera
            this.targetGaze.set(0, 1.38, 0.92);
        }

        // Smoothly lerp the Object3D's position — three-vrm calls .getWorldPosition() on it
        this.currentGaze.lerp(this.targetGaze, Math.min(delta * 5.0, 1.0));
        this.lookAtTarget.position.copy(this.currentGaze);
    }

    /**
     * Subtle organic idle breathing & active listening head tilts/nods
     */
    updateProceduralBones(elapsedTime, delta) {
        if (!this.vrm || !this.vrm.humanoid) return;
        const humanoid = this.vrm.humanoid;

        // 1. Natural Breathing (Chest & Spine gentle sine oscillation)
        const spineNode = humanoid.getNormalizedBoneNode('spine');
        const chestNode = humanoid.getNormalizedBoneNode('chest');
        const headNode = humanoid.getNormalizedBoneNode('head');
        const neckNode = humanoid.getNormalizedBoneNode('neck');

        const breathCycle = Math.sin(elapsedTime * 1.5); // ~0.24 Hz natural breath
        if (spineNode) {
            spineNode.rotation.x = breathCycle * 0.012;
        }
        if (chestNode) {
            chestNode.rotation.x = breathCycle * 0.018;
        }

        // 2. Head Tilt & Micro-nods depending on behavioral state
        if (headNode) {
            let targetTiltZ = 0; // sideways tilt (curiosity)
            let targetPitchX = 0; // nod (agreement)

            if (this.state === 'listening') {
                // Cute head tilt when attentively listening
                targetTiltZ = 0.06; // slight curious tilt
                // Responsive micro-nodding
                if (this.isNodding) {
                    this.nodTime += delta * 7.0;
                    targetPitchX = Math.sin(this.nodTime) * 0.08;
                    if (this.nodTime >= Math.PI * 2) {
                        this.isNodding = false;
                        this.nodTime = 0;
                    }
                } else if (Math.random() < 0.008) {
                    // Occasional spontaneous nod
                    this.isNodding = true;
                    this.nodTime = 0;
                }
            } else if (this.state === 'thinking') {
                targetTiltZ = -0.05;
                targetPitchX = -0.04; // slight tilt back
            } else if (this.state === 'speaking') {
                // Natural speech cadence head bobbing
                targetPitchX = Math.sin(elapsedTime * 4.0) * 0.035 * (this.currentMouthOpen + 0.1);
                targetTiltZ = Math.cos(elapsedTime * 2.5) * 0.025;
            }

            this.currentHeadTilt = THREE.MathUtils.lerp(this.currentHeadTilt, targetTiltZ, Math.min(delta * 4.0, 1.0));
            headNode.rotation.z = this.currentHeadTilt;
            headNode.rotation.x = THREE.MathUtils.lerp(headNode.rotation.x, targetPitchX, Math.min(delta * 6.0, 1.0));
        }
    }

    /**
     * Behavioral state switcher: 'idle' | 'listening' | 'thinking' | 'speaking'
     */
    setState(newState) {
        if (this.state === newState) return;
        this.state = newState;

        if (!this.vrm || !this.vrm.expressionManager) return;
        const expressions = this.vrm.expressionManager;

        // Reset emotional blends smoothly
        if (newState === 'listening') {
            // Perk up: welcoming smile + curious alert eyes
            expressions.setValue('happy', 0.28);
            expressions.setValue('relaxed', 0.15);
            expressions.setValue('surprised', 0.05);
            this.isNodding = true;
            this.nodTime = 0;
        } else if (newState === 'thinking') {
            expressions.setValue('happy', 0.08);
            expressions.setValue('relaxed', 0.25);
            expressions.setValue('surprised', 0.0);
        } else if (newState === 'speaking') {
            expressions.setValue('happy', 0.32);
            expressions.setValue('relaxed', 0.10);
            expressions.setValue('surprised', 0.0);
        } else {
            // Idle
            expressions.setValue('happy', 0.20);
            expressions.setValue('relaxed', 0.10);
            expressions.setValue('surprised', 0.0);
        }
    }

    /**
     * Triggered on user barge-in / interruption
     */
    bargeIn() {
        this.setState('listening');
        this.currentMouthOpen = 0;
        this.currentVowelI = 0;
        this.currentVowelO = 0;
        if (this.vrm && this.vrm.expressionManager) {
            this.vrm.expressionManager.setValue('aa', 0);
            this.vrm.expressionManager.setValue('ih', 0);
            this.vrm.expressionManager.setValue('oh', 0);
            this.vrm.expressionManager.update();
        }
        // Immediate receptive nod
        this.isNodding = true;
        this.nodTime = 0;
    }

    /**
     * Trigger micro-nod when user voice energy detected
     */
    triggerUserSpeechNod() {
        if (this.state === 'listening' && !this.isNodding) {
            this.isNodding = true;
            this.nodTime = 0;
        }
    }

    /**
     * Main Render Loop (60 FPS)
     */
    animate() {
        requestAnimationFrame(this.animate);

        const delta = this.clock.getDelta();
        const elapsedTime = this.clock.getElapsedTime();

        if (this.vrm) {
            this.updateAutoBlink(delta);
            this.updateLookAt(delta);
            this.updateLipSync(delta);
            this.updateProceduralBones(elapsedTime, delta);

            // Update Spring Bone physics (hair sway, clothing bounce)
            this.vrm.update(delta);
        }

        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }

    dispose() {
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
        }
        if (this.renderer) {
            this.renderer.dispose();
        }
        if (this.vrm) {
            VRMUtils.deepDispose(this.vrm.scene);
        }
    }
}
