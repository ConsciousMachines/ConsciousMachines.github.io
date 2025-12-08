// Tile dimensions (per tile)
const HP = 128;
const WP = 128;

// Maximum grid size
const MAX_HG = 4;
const MAX_WG = 6;

// Circular buffer settings
const BUFFER_SIZE = 3;

// Current grid size (will be calculated dynamically)
let HG = 4;
let WG = 6;
let g_tex_H = HP * HG;
let g_tex_W = WP * WG;

let worker;
let isReady = false;
let animationRunning = false;

// WebGL variables
let gl;
let program;
let vao;
let textures = [];
let pbos = [];

// Animation state
let currTexIdx = 0;
let nextTexIdx = 1;
let t = 0.0;

// Blocking queue
let imageQueue = [];
let queueWaiters = [];
const MAX_QUEUE_SIZE = BUFFER_SIZE;

// Progressive upload state
let pendingUpload = null;

// FPS tracking
let frameCount = 0;
let lastFpsUpdate = performance.now();
let fps = 0;
let lastFrameTime = performance.now();
let frameTimeAvg = 0;
let workerTimeAvg = 0;

// PCA Parameters (controllable by sliders)
let pcaParams = {
    k: 0.9,
    s: 0.9,
    p: 1.0,
    num_pc: 84
};

let steps = 10;
const dt_base = 1.0;
let dt = dt_base / steps;

// Vertex shader
const vertexShaderSource = `#version 300 es
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aUV;
out vec2 uv;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    uv = aUV;
}
`;

// Fragment shader
const fragmentShaderSource = `#version 300 es
precision highp float;
in vec2 uv;
out vec4 FragColor;

uniform sampler2D texA;
uniform sampler2D texB;
uniform float t;

void main() {
    vec3 colorA = texture(texA, uv).rgb;
    vec3 colorB = texture(texB, uv).rgb;
    vec3 result = mix(colorA, colorB, t);
    FragColor = vec4(result, 1.0);
}
`;

// Initialize sliders
function initSliders() {
    // K slider
    const kSlider = document.getElementById('k-slider');
    const kValue = document.getElementById('k-value');
    kSlider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value) / 100;
        pcaParams.k = value;
        kValue.textContent = value.toFixed(2);
        updateWorkerParams();
    });
    
    // S slider
    const sSlider = document.getElementById('s-slider');
    const sValue = document.getElementById('s-value');
    sSlider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value) / 100;
        pcaParams.s = value;
        sValue.textContent = value.toFixed(2);
        updateWorkerParams();
    });
    
    // Steps slider
    const stepsSlider = document.getElementById('steps-slider');
    const stepsValue = document.getElementById('steps-value');
    stepsSlider.addEventListener('input', (e) => {
        steps = parseInt(e.target.value);
        dt = dt_base / steps;
        stepsValue.textContent = steps;
        // No need to update worker for steps
    });
    
    // P slider
    const pSlider = document.getElementById('p-slider');
    const pValue = document.getElementById('p-value');
    pSlider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value) / 100;
        pcaParams.p = value;
        pValue.textContent = value.toFixed(2);
        updateWorkerParams();
    });
    
    // Num PC slider
    const numPcSlider = document.getElementById('num_pc-slider');
    const numPcValue = document.getElementById('num_pc-value');
    numPcSlider.addEventListener('input', (e) => {
        pcaParams.num_pc = parseInt(e.target.value);
        numPcValue.textContent = pcaParams.num_pc;
        updateWorkerParams();
    });
}

// Update worker with new parameters
function updateWorkerParams() {
    if (worker && isReady) {
        worker.postMessage({
            type: 'update_params',
            params: pcaParams
        });
    }
}

// Calculate optimal grid size based on screen dimensions
function calculateGridSize() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    
    // Account for UI elements
    const headerHeight = document.querySelector('h1')?.offsetHeight || 50;
    const statsHeight = document.querySelector('.stats')?.offsetHeight || 40;
    const controlsHeight = document.querySelector('.controls-panel')?.offsetHeight || 100;
    const uiHeight = headerHeight + statsHeight + controlsHeight + 20; // 20px extra padding
    
    const availableHeight = height - uiHeight;
    
    // Calculate how many tiles can fit
    const maxTilesW = Math.floor(width / WP);
    const maxTilesH = Math.floor(availableHeight / HP);
    
    // Clamp to maximum
    const newWG = Math.min(maxTilesW, MAX_WG);
    const newHG = Math.min(maxTilesH, MAX_HG);
    
    // Ensure at least 1x1
    return {
        HG: Math.max(1, newHG),
        WG: Math.max(1, newWG)
    };
}

// Calculate canvas size to fit screen while maintaining aspect ratio
function calculateCanvasSize(hg, wg) {
    const idealWidth = wg * WP;
    const idealHeight = hg * HP;
    
    const screenWidth = window.innerWidth;
    const screenHeight = window.innerHeight;
    
    // Account for UI elements
    const headerHeight = document.querySelector('h1')?.offsetHeight || 50;
    const statsHeight = document.querySelector('.stats')?.offsetHeight || 40;
    const controlsHeight = document.querySelector('.controls-panel')?.offsetHeight || 100;
    const uiHeight = headerHeight + statsHeight + controlsHeight + 20; // 20px extra padding
    
    const availableHeight = screenHeight - uiHeight;
    
    // Scale to fit screen
    const scaleW = screenWidth / idealWidth;
    const scaleH = availableHeight / idealHeight;
    const scale = Math.min(scaleW, scaleH, 1.0); // Don't scale up
    
    return {
        width: Math.floor(idealWidth * scale),
        height: Math.floor(idealHeight * scale)
    };
}

// Compile shader
function compileShader(gl, source, type) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('Shader compile error:', gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
    }
    
    return shader;
}

// Create shader program
function createProgram(gl, vsSource, fsSource) {
    const vs = compileShader(gl, vsSource, gl.VERTEX_SHADER);
    const fs = compileShader(gl, fsSource, gl.FRAGMENT_SHADER);
    
    const program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error('Program link error:', gl.getProgramInfoLog(program));
        return null;
    }
    
    return program;
}

// Initialize WebGL
function initWebGL() {
    const canvas = document.getElementById('canvas');
    
    // Calculate optimal size
    const size = calculateCanvasSize(HG, WG);
    canvas.width = g_tex_W;
    canvas.height = g_tex_H;
    canvas.style.width = size.width + 'px';
    canvas.style.height = size.height + 'px';
    
    gl = canvas.getContext('webgl2');
    if (!gl) {
        alert('WebGL 2 not supported');
        return false;
    }
    
    console.log('WebGL Version:', gl.getParameter(gl.VERSION));
    console.log(`Grid: ${HG}x${WG}, Texture: ${g_tex_W}x${g_tex_H}, Display: ${size.width}x${size.height}`);
    
    // Compile shaders
    program = createProgram(gl, vertexShaderSource, fragmentShaderSource);
    
    // Create fullscreen quad
    const quad = new Float32Array([
         1.0,  1.0,  1.0, 0.0,
         1.0, -1.0,  1.0, 1.0,
        -1.0, -1.0,  0.0, 1.0,
        -1.0,  1.0,  0.0, 0.0
    ]);
    
    const indices = new Uint32Array([0, 1, 3, 1, 2, 3]);
    
    // Create VAO
    vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    
    // Create VBO
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
    
    // Create EBO
    const ebo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
    
    // Position attribute
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 4 * 4, 0);
    gl.enableVertexAttribArray(0);
    
    // UV attribute
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 4 * 4, 2 * 4);
    gl.enableVertexAttribArray(1);
    
    // Create circular buffer of textures
    for (let i = 0; i < BUFFER_SIZE; i++) {
        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, g_tex_W, g_tex_H, 0, gl.RGB, gl.UNSIGNED_BYTE, null);
        textures.push(texture);
    }
    
    // Create PBOs
    const pboSize = g_tex_W * g_tex_H * 3;
    for (let i = 0; i < BUFFER_SIZE; i++) {
        const pbo = gl.createBuffer();
        gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, pbo);
        gl.bufferData(gl.PIXEL_UNPACK_BUFFER, pboSize, gl.STREAM_DRAW);
        pbos.push(pbo);
    }
    
    gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindVertexArray(null);
    
    return true;
}

// Cleanup WebGL resources
function cleanupWebGL() {
    if (!gl) return;
    
    // Delete textures
    for (let tex of textures) {
        gl.deleteTexture(tex);
    }
    textures = [];
    
    // Delete PBOs
    for (let pbo of pbos) {
        gl.deleteBuffer(pbo);
    }
    pbos = [];
}

// Blocking queue: put
function queuePut(item) {
    imageQueue.push(item);
    
    if (queueWaiters.length > 0) {
        const resolve = queueWaiters.shift();
        const nextItem = imageQueue.shift();
        resolve(nextItem);
    }
    
    if (imageQueue.length < MAX_QUEUE_SIZE) {
        worker.postMessage({ type: 'queue_has_space' });
    }
}

// Blocking queue: get
function queueGet() {
    return new Promise((resolve) => {
        if (imageQueue.length > 0) {
            resolve(imageQueue.shift());
            if (imageQueue.length < MAX_QUEUE_SIZE) {
                worker.postMessage({ type: 'queue_has_space' });
            }
        } else {
            queueWaiters.push(resolve);
        }
    });
}

// Convert RGBA to RGB
function convertRGBAtoRGB(rgbaData) {
    const rgbData = new Uint8Array(g_tex_W * g_tex_H * 3);
    for (let i = 0; i < g_tex_W * g_tex_H; i++) {
        rgbData[i * 3] = rgbaData[i * 4];
        rgbData[i * 3 + 1] = rgbaData[i * 4 + 1];
        rgbData[i * 3 + 2] = rgbaData[i * 4 + 2];
    }
    return rgbData;
}

// Upload entire texture
function uploadTextureComplete(slotIdx, imageData) {
    const rgbData = convertRGBAtoRGB(imageData);
    gl.bindTexture(gl.TEXTURE_2D, textures[slotIdx]);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, g_tex_W, g_tex_H, gl.RGB, gl.UNSIGNED_BYTE, rgbData);
    gl.bindTexture(gl.TEXTURE_2D, null);
}

// Upload chunk with PBO
function uploadTextureChunkWithPBO(texIdx, pboIdx, rgbData, yOffset, chunkHeight) {
    const rowBytes = g_tex_W * 3;
    const startByte = yOffset * rowBytes;
    const chunkBytes = chunkHeight * rowBytes;
    const chunkData = rgbData.subarray(startByte, startByte + chunkBytes);
    
    gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, pbos[pboIdx]);
    gl.bufferSubData(gl.PIXEL_UNPACK_BUFFER, startByte, chunkData);
    
    gl.bindTexture(gl.TEXTURE_2D, textures[texIdx]);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, yOffset, g_tex_W, chunkHeight, gl.RGB, gl.UNSIGNED_BYTE, startByte);
    
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, null);
}

// Process progressive upload
function processProgressiveUpload() {
    if (!pendingUpload) return;
    
    const { slotIdx, pboIdx, rgbData, rowsUploaded } = pendingUpload;
    
    const rowsLeft = g_tex_H - rowsUploaded;
    const framesLeft = Math.max(1, Math.floor((1.0 - t) / dt));
    const rowsToUpload = Math.min(rowsLeft, Math.floor(rowsLeft / framesLeft) + 1);
    
    uploadTextureChunkWithPBO(slotIdx, pboIdx, rgbData, rowsUploaded, rowsToUpload);
    
    pendingUpload.rowsUploaded += rowsToUpload;
    
    if (pendingUpload.rowsUploaded >= g_tex_H) {
        pendingUpload = null;
    }
}

// Render
function render() {
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    
    gl.useProgram(program);
    
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, textures[currTexIdx]);
    const texALoc = gl.getUniformLocation(program, 'texA');
    gl.uniform1i(texALoc, 0);
    
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, textures[nextTexIdx]);
    const texBLoc = gl.getUniformLocation(program, 'texB');
    gl.uniform1i(texBLoc, 1);
    
    const tLoc = gl.getUniformLocation(program, 't');
    gl.uniform1f(tLoc, t);
    
    gl.bindVertexArray(vao);
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_INT, 0);
    gl.bindVertexArray(null);
}

// Update FPS
function updateFPS() {
    frameCount++;
    const now = performance.now();
    const elapsed = now - lastFpsUpdate;
    
    const currentFrameTime = now - lastFrameTime;
    frameTimeAvg = frameTimeAvg * 0.9 + currentFrameTime * 0.1;
    lastFrameTime = now;
    
    if (elapsed >= 1000) {
        fps = Math.round((frameCount * 1000) / elapsed);
        document.getElementById('fps').textContent = 
            `FPS: ${fps} | Frame: ${frameTimeAvg.toFixed(1)}ms | Worker: ${workerTimeAvg.toFixed(1)}ms | Grid: ${HG}x${WG}`;
        frameCount = 0;
        lastFpsUpdate = now;
    }
}

// Modify the animate function to remove progress bar update
async function animate() {
    if (!animationRunning) {
        requestAnimationFrame(animate);
        return;
    }
    
    // Step 1: Progressive upload
    if (pendingUpload) {
        processProgressiveUpload();
        // REMOVED: updateProgressBar();
    }
    
    // Step 2: Transition check
    if (t > 1.0) {
        t = 0.0;
        currTexIdx = (currTexIdx + 1) % BUFFER_SIZE;
        nextTexIdx = (nextTexIdx + 1) % BUFFER_SIZE;
        
        if (pendingUpload !== null) {
            console.error('❌ ERROR: Pending upload not finished!');
        }
        
        const { slotIdx, imageData, generationTime } = await queueGet();
        workerTimeAvg = workerTimeAvg * 0.9 + generationTime * 0.1;
        
        const rgbData = convertRGBAtoRGB(new Uint8ClampedArray(imageData));
        
        pendingUpload = {
            slotIdx,
            pboIdx: slotIdx,
            rgbData,
            rowsUploaded: 0
        };
    }
    
    // Step 3: Render
    render();
    updateFPS();
    
    // Step 4: Increment t
    t += dt;
    
    requestAnimationFrame(animate);
}

// Initialize worker
function initWorker(hg, wg) {
    if (worker) {
        worker.terminate();
    }
    
    worker = new Worker('worker.js');
    isReady = false;
    
    worker.onmessage = function(e) {
        const { type, slotIdx, data, generationTime } = e.data;
        
        switch(type) {
            case 'ready':
                isReady = true;
                console.log(`Worker ready with grid ${hg}x${wg}`);
                worker.postMessage({ type: 'start_generating' });
                fillBuffer();
                break;
                
            case 'image':
                queuePut({ slotIdx, imageData: data, generationTime });
                break;
                
            case 'error':
                console.error('Worker error:', data);
                alert('Error: ' + data);
                break;
        }
    };
    
    worker.onerror = function(error) {
        console.error('Worker error:', error);
    };
    
    // Send grid size to worker
    worker.postMessage({ type: 'set_grid', HG: hg, WG: wg });
}

// Fill buffer
async function fillBuffer() {
    console.log('Filling buffer...');
    
    for (let i = 0; i < BUFFER_SIZE; i++) {
        const { slotIdx, imageData, generationTime } = await queueGet();
        uploadTextureComplete(slotIdx, new Uint8ClampedArray(imageData));
        console.log(`Buffer filled: ${i + 1}/${BUFFER_SIZE} (${generationTime.toFixed(1)}ms)`);
    }
    
    console.log('Buffer full, starting animation...');
    animationRunning = true;
}

// Restart with new grid size
async function restartWithGridSize(newHG, newWG) {
    console.log(`Restarting with grid ${newHG}x${newWG}...`);
    
    // Stop animation
    animationRunning = false;
    await new Promise(resolve => setTimeout(resolve, 100));
    
    // Clear queue
    imageQueue = [];
    queueWaiters = [];
    pendingUpload = null;
    t = 0.0;
    currTexIdx = 0;
    nextTexIdx = 1;
    
    // Update dimensions
    HG = newHG;
    WG = newWG;
    g_tex_H = HP * HG;
    g_tex_W = WP * WG;
    
    // Cleanup and reinit WebGL
    cleanupWebGL();
    initWebGL();
    
    // Restart worker
    initWorker(HG, WG);
}

// Handle resize
let resizeTimeout;
function handleResize() {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        const newGrid = calculateGridSize();
        
        // Check if grid size changed
        if (newGrid.HG !== HG || newGrid.WG !== WG) {
            console.log(`Grid size changed: ${HG}x${WG} → ${newGrid.HG}x${newGrid.WG}`);
            restartWithGridSize(newGrid.HG, newGrid.WG);
        } else {
            // Just resize canvas display
            const canvas = document.getElementById('canvas');
            const size = calculateCanvasSize(HG, WG);
            canvas.style.width = size.width + 'px';
            canvas.style.height = size.height + 'px';
        }
    }, 500); // Debounce
}

// Modify init function to initialize sliders
function init() {
    // Initialize sliders first
    initSliders();
    
    // Calculate initial grid size
    const grid = calculateGridSize();
    HG = grid.HG;
    WG = grid.WG;
    g_tex_H = HP * HG;
    g_tex_W = WP * WG;
    
    console.log(`Initial grid: ${HG}x${WG}`);
    
    if (initWebGL()) {
        initWorker(HG, WG);
        animate();
        
        // Add resize listener
        window.addEventListener('resize', handleResize);
        window.addEventListener('orientationchange', handleResize);
    } else {
        alert('Failed to initialize WebGL');
    }
}

init();