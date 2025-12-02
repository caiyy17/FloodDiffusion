/**
 * Main application logic
 * Handles UI interactions, API calls, and 3D rendering loop
 */

class MotionApp {
    constructor() {
        this.isRunning = false;
        this.targetFps = 20; // Model generates data at 20fps
        this.frameInterval = 1000 / this.targetFps; // 50ms
        this.lastFetchTime = 0;
        this.frameCount = 0;
        this.fpsCounter = 0;
        this.fpsUpdateTime = 0;
        this.lastRenderTime = 0;
        
        // Session management
        this.sessionId = this.generateSessionId();
        
        // Camera follow settings
        this.lastUserInteraction = 0;
        this.autoFollowDelay = 2000; // Auto-follow after 2 seconds of inactivity (reduced from 3s)
        this.currentRootPos = new THREE.Vector3(0, 1, 0);
        
        this.initThreeJS();
        this.initUI();
        this.updateStatus();
        this.setupBeforeUnload();
        
        console.log('Session ID:', this.sessionId);
    }
    
    generateSessionId() {
        // Generate a simple unique session ID
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    setupBeforeUnload() {
        // Handle page close/refresh - send reset request
        window.addEventListener('beforeunload', () => {
            // Send synchronous reset if we're generating
            if (!this.isIdle) {
                // Use Blob to set correct Content-Type for JSON
                const blob = new Blob(
                    [JSON.stringify({session_id: this.sessionId})],
                    {type: 'application/json'}
                );
                navigator.sendBeacon('/api/reset', blob);
                console.log('Sent reset beacon on page unload');
            }
        });
        
        // Also handle visibility change (tab hidden, mobile app switch)
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && !this.isIdle && this.isRunning) {
                // User switched away while generating - they might not come back
                // Note: Don't reset immediately, let the frame consumption monitor handle it
                console.log('Tab hidden while generating - consumption monitor will auto-reset if needed');
            }
        });
    }
    
    initThreeJS() {
        // Get canvas
        const canvas = document.getElementById('renderCanvas');
        const container = document.getElementById('canvas-container');
        
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xffffff);  // White background
        
        // Create camera
        this.camera = new THREE.PerspectiveCamera(
            60,
            container.clientWidth / container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(3, 1.5, 3);
        this.camera.lookAt(0, 1, 0);
        
        // Create renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: canvas,
            antialias: true
        });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;
        
        // Add lights - bright and soft
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
        this.scene.add(ambientLight);
        
        const keyLight = new THREE.DirectionalLight(0xffffff, 0.8);
        keyLight.position.set(5, 8, 3);
        keyLight.castShadow = true;
        keyLight.shadow.mapSize.width = 2048;
        keyLight.shadow.mapSize.height = 2048;
        keyLight.shadow.camera.near = 0.5;
        keyLight.shadow.camera.far = 50;
        keyLight.shadow.camera.left = -5;
        keyLight.shadow.camera.right = 5;
        keyLight.shadow.camera.top = 5;
        keyLight.shadow.camera.bottom = -5;
        keyLight.shadow.bias = -0.0001;
        this.scene.add(keyLight);
        
        // Fill light
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.4);
        fillLight.position.set(-3, 5, -3);
        this.scene.add(fillLight);
        
        // Add ground plane - light gray, very large
        const groundGeometry = new THREE.PlaneGeometry(1000, 1000);
        const groundMaterial = new THREE.ShadowMaterial({
            opacity: 0.15
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = 0;
        ground.receiveShadow = true;
        this.scene.add(ground);
        
        // Add infinite-looking grid - very large grid
        const gridHelper = new THREE.GridHelper(1000, 1000, 0xdddddd, 0xeeeeee);
        gridHelper.position.y = 0.01;
        this.scene.add(gridHelper);
        
        // Add orbit controls
        this.controls = new THREE.OrbitControls(this.camera, canvas);
        this.controls.target.set(0, 1, 0);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.update();
        
        // Listen for user interaction - record time
        const updateInteractionTime = () => {
            this.lastUserInteraction = Date.now();
        };
        canvas.addEventListener('mousedown', updateInteractionTime);
        canvas.addEventListener('wheel', updateInteractionTime);
        canvas.addEventListener('touchstart', updateInteractionTime);
        
        // Create skeleton
        this.skeleton = new Skeleton3D(this.scene);
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
        
        // Start render loop
        this.animate();
    }
    
    initUI() {
        // Get UI elements
        this.motionText = document.getElementById('motionText');
        this.historyLength = document.getElementById('historyLength');
        this.startResetBtn = document.getElementById('startResetBtn');
        this.updateBtn = document.getElementById('updateBtn');
        this.pauseResumeBtn = document.getElementById('pauseResumeBtn');
        this.statusEl = document.getElementById('status');
        this.bufferSizeEl = document.getElementById('bufferSize');
        this.fpsEl = document.getElementById('fps');
        this.frameCountEl = document.getElementById('frameCount');
        this.conflictWarning = document.getElementById('conflictWarning');
        this.forceTakeoverBtn = document.getElementById('forceTakeoverBtn');
        this.cancelTakeoverBtn = document.getElementById('cancelTakeoverBtn');
        
        // Track state
        this.isPaused = false;
        this.isIdle = true;
        this.isProcessing = false;  // Prevent concurrent API calls
        this.pendingStartRequest = null;  // Store pending start request data
        
        // Attach event listeners
        this.startResetBtn.addEventListener('click', () => this.toggleStartReset());
        this.updateBtn.addEventListener('click', () => this.updateText());
        this.pauseResumeBtn.addEventListener('click', () => this.togglePauseResume());
        this.forceTakeoverBtn.addEventListener('click', () => this.handleForceTakeover());
        this.cancelTakeoverBtn.addEventListener('click', () => this.handleCancelTakeover());
    }
    
    async toggleStartReset() {
        if (this.isProcessing) return;  // Prevent concurrent operations
        
        if (this.isIdle) {
            // Currently idle, so start
            await this.startGeneration();
        } else {
            // Currently running/paused, so reset
            await this.reset();
        }
    }
    
    async startGeneration(force = false) {
        if (this.isProcessing) return;  // Prevent concurrent operations
        
        const text = this.motionText.value.trim();
        if (!text) {
            alert('Please enter a motion description');
            return;
        }
        
        const historyLength = parseInt(this.historyLength.value) || 30;
        if (historyLength < 10 || historyLength > 200) {
            alert('History length must be between 10 and 200');
            return;
        }
        
        this.isProcessing = true;
        this.statusEl.textContent = 'Initializing...';
        
        try {
            const response = await fetch('/api/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    session_id: this.sessionId,
                    text: text,
                    history_length: historyLength,
                    force: force
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.isRunning = true;
                this.isPaused = false;
                this.isIdle = false;
                this.frameCount = 0;
                this.startResetBtn.textContent = 'Reset';
                this.startResetBtn.classList.remove('btn-primary');
                this.startResetBtn.classList.add('btn-danger');
                this.updateBtn.disabled = false;
                this.pauseResumeBtn.disabled = false;
                this.pauseResumeBtn.textContent = 'Pause';
                this.statusEl.textContent = 'Running';
                this.startFrameLoop();
            } else if (response.status === 409 && data.conflict) {
                // Another session is running, show warning UI
                this.statusEl.textContent = 'Conflict - Another user is generating';
                this.conflictWarning.style.display = 'block';
                
                // Store request data for later
                this.pendingStartRequest = {
                    text: text,
                    history_length: historyLength
                };
                
                return;
            } else {
                // Other errors
                alert('Error: ' + data.message);
                this.statusEl.textContent = 'Idle';
                this.isIdle = true;
                this.isRunning = false;
                this.isPaused = false;
            }
        } catch (error) {
            console.error('Error starting generation:', error);
            alert('Failed to start generation: ' + error.message);
            this.statusEl.textContent = 'Idle';
            // Keep idle state on error
            this.isIdle = true;
            this.isRunning = false;
            this.isPaused = false;
        } finally {
            this.isProcessing = false;
        }
    }
    
    async updateText() {
        if (this.isProcessing) return;  // Prevent concurrent operations
        
        const text = this.motionText.value.trim();
        if (!text) {
            alert('Please enter a motion description');
            return;
        }
        
        this.isProcessing = true;
        try {
            const response = await fetch('/api/update_text', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    session_id: this.sessionId,
                    text: text
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                console.log('Text updated:', text);
            } else {
                alert('Error: ' + data.message);
            }
        } catch (error) {
            console.error('Error updating text:', error);
        } finally {
            this.isProcessing = false;
        }
    }
    
    async togglePauseResume() {
        if (this.isProcessing) return;  // Prevent concurrent operations
        if (this.isPaused) {
            // Currently paused, so resume
            await this.resumeGeneration();
        } else {
            // Currently running, so pause
            await this.pauseGeneration();
        }
    }
    
    async pauseGeneration() {
        this.isProcessing = true;
        try {
            const response = await fetch('/api/pause', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session_id: this.sessionId})
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.isRunning = false;
                this.isPaused = true;
                this.pauseResumeBtn.textContent = 'Resume';
                this.pauseResumeBtn.classList.remove('btn-warning');
                this.pauseResumeBtn.classList.add('btn-success');
                this.updateBtn.disabled = true;
                this.statusEl.textContent = 'Paused';
                console.log('Generation paused (state preserved)');
            }
        } catch (error) {
            console.error('Error pausing generation:', error);
        } finally {
            this.isProcessing = false;
        }
    }
    
    async resumeGeneration() {
        this.isProcessing = true;
        try {
            const response = await fetch('/api/resume', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session_id: this.sessionId})
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.isRunning = true;
                this.isPaused = false;
                this.pauseResumeBtn.textContent = 'Pause';
                this.pauseResumeBtn.classList.remove('btn-success');
                this.pauseResumeBtn.classList.add('btn-warning');
                this.updateBtn.disabled = false;
                this.statusEl.textContent = 'Running';
                this.startFrameLoop();
                console.log('Generation resumed');
            }
        } catch (error) {
            console.error('Error resuming generation:', error);
        } finally {
            this.isProcessing = false;
        }
    }
    
    async reset() {
        if (this.isProcessing) return;  // Prevent concurrent operations
        
        const historyLength = parseInt(this.historyLength.value) || 30;
        
        this.isProcessing = true;
        try {
            const response = await fetch('/api/reset', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    session_id: this.sessionId,
                    history_length: historyLength
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.isRunning = false;
                this.isPaused = false;
                this.isIdle = true;
                this.frameCount = 0;
                this.startResetBtn.textContent = 'Start';
                this.startResetBtn.classList.remove('btn-danger');
                this.startResetBtn.classList.add('btn-primary');
                this.updateBtn.disabled = true;
                this.pauseResumeBtn.disabled = true;
                this.pauseResumeBtn.textContent = 'Pause';
                this.pauseResumeBtn.classList.remove('btn-success');
                this.pauseResumeBtn.classList.add('btn-warning');
                this.statusEl.textContent = 'Idle';
                this.bufferSizeEl.textContent = '0 / 4';
                this.frameCountEl.textContent = '0';
                
                // Clear trail
                if (this.skeleton) {
                    this.skeleton.clearTrail();
                }
                
                console.log('Reset complete - all state cleared');
            }
        } catch (error) {
            console.error('Error resetting:', error);
        } finally {
            this.isProcessing = false;
        }
    }
    
    async handleForceTakeover() {
        // Hide warning
        this.conflictWarning.style.display = 'none';
        
        if (!this.pendingStartRequest) return;
        
        // Retry with force=true
        this.isProcessing = false;
        await this.startGeneration(true);
        
        this.pendingStartRequest = null;
    }
    
    handleCancelTakeover() {
        // Hide warning
        this.conflictWarning.style.display = 'none';
        this.statusEl.textContent = 'Idle';
        this.isProcessing = false;
        this.pendingStartRequest = null;
    }
    
    startFrameLoop() {
        this.lastFetchTime = performance.now();
        this.fetchFrame();
    }
    
    fetchFrame() {
        if (!this.isRunning) return;
        
        const now = performance.now();
        const elapsed = now - this.lastFetchTime;
        
        // Only fetch when frame interval is reached
        if (elapsed >= this.frameInterval) {
            this.lastFetchTime = now;
            
            fetch(`/api/get_frame?session_id=${this.sessionId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        this.skeleton.updatePose(data.joints);
                        this.frameCount++;
                        this.frameCountEl.textContent = this.frameCount;
                        
                        // Update current root position
                        this.currentRootPos.set(
                            data.joints[0][0],
                            data.joints[0][1],
                            data.joints[0][2]
                        );
                        
                        // Auto-follow (if user hasn't interacted for a while)
                        this.updateAutoFollow();
                    }
                })
                .catch(error => {
                    console.error('Error fetching frame:', error);
                });
        }
        
        // Use requestAnimationFrame for continuous checking
        requestAnimationFrame(() => this.fetchFrame());
    }
    
    updateAutoFollow() {
        const timeSinceInteraction = Date.now() - this.lastUserInteraction;
        
        // Auto-follow if user hasn't interacted for more than 3 seconds
        if (timeSinceInteraction > this.autoFollowDelay) {
            // Calculate camera offset relative to current target
            const currentOffset = new THREE.Vector3().subVectors(
                this.camera.position, 
                this.controls.target
            );
            
            // New target position (character position, waist height)
            const newTarget = this.currentRootPos.clone();
            newTarget.y = 1.0;
            
            // Calculate new camera position (maintain relative offset)
            const newCameraPos = newTarget.clone().add(currentOffset);
            
            // Smooth interpolation follow (increased lerp factor for more obvious following)
            // 0.2 = more aggressive following, 0.05 = gentle following
            this.controls.target.lerp(newTarget, 0.2);
            this.camera.position.lerp(newCameraPos, 0.2);
            
            // Debug log (comment out in production)
            // console.log('Auto-follow active, tracking:', newTarget);
        }
    }
    
    async updateStatus() {
        try {
            const response = await fetch(`/api/status?session_id=${this.sessionId}`);
            const data = await response.json();
            
            if (data.initialized) {
                this.bufferSizeEl.textContent = `${data.buffer_size} / ${data.target_size}`;
            }
        } catch (error) {
            // Silently fail for status updates
        }
        
        // Update status every 500ms
        setTimeout(() => this.updateStatus(), 500);
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Update controls
        this.controls.update();
        
        // Update FPS counter
        const now = performance.now();
        this.fpsCounter++;
        if (now - this.fpsUpdateTime > 1000) {
            this.fpsEl.textContent = this.fpsCounter;
            this.fpsCounter = 0;
            this.fpsUpdateTime = now;
        }
        
        // Render scene
        this.renderer.render(this.scene, this.camera);
    }
    
    onWindowResize() {
        const container = document.getElementById('canvas-container');
        this.camera.aspect = container.clientWidth / container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(container.clientWidth, container.clientHeight);
    }
}

// Initialize app when page loads
window.addEventListener('DOMContentLoaded', () => {
    window.app = new MotionApp();
});

