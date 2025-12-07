// Canvas dimensions
const HP = 128;
const WP = 128;
const HG = 4;
const WG = 6;
const g_tex_H = HP * HG;
const g_tex_W = WP * WG;

// Circular buffer settings
const BUFFER_SIZE = 3;

let worker;
let isReady = false;

// WebGL variables
let gl;
let program;
let vao;
let textures = [];  // Circular buffer of textures
let pbos = [];      // Pixel Buffer Objects for async uploads

// Animation state
let currTexIdx = 0;
let nextTexIdx = 1;
let t = 0.0;
const steps = 10;  // Increased from 5 to 10
const dt = 1.0 / steps;

// Blocking queue implementation
let imageQueue = [];
let queueWaiters = [];  // Waiting promises for get()
const MAX_QUEUE_SIZE = BUFFER_SIZE;

// Progressive upload state
let pendingUpload = null;  // { slotIdx, pboIdx, rgbData, rowsUploaded }

// FPS and performance tracking
let frameCount = 0;
let lastFpsUpdate = performance.now();
let fps = 0;
let lastFrameTime = performance.now();
let frameTimeAvg = 0;
let workerTimeAvg = 0;

// Vertex shader - simple passthrough
const vertexShaderSource = `#version 300 es
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aUV;
out vec2 uv;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    uv = aUV;
}
`;

// Fragment shader - interpolate between two textures
const fragmentShaderSource = `#version 300 es
precision highp float;
in vec2 uv;
out vec4 FragColor;

uniform sampler2D texA;  // Current texture
uniform sampler2D texB;  // Next texture
uniform float t;         // Interpolation [0, 1]

void main() {
    vec3 colorA = texture(texA, uv).rgb;
    vec3 colorB = texture(texB, uv).rgb;
    
    // Interpolate
    vec3 result = mix(colorA, colorB, t);
    FragColor = vec4(result, 1.0);
}
`;

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
    canvas.width = g_tex_W;
    canvas.height = g_tex_H;
    
    gl = canvas.getContext('webgl2');
    if (!gl) {
        alert('WebGL 2 not supported');
        return false;
    }
    
    console.log('WebGL Version:', gl.getParameter(gl.VERSION));
    
    // Compile shaders
    program = createProgram(gl, vertexShaderSource, fragmentShaderSource);
    
    // Create fullscreen quad
    const quad = new Float32Array([
        // pos(x,y), uv(u,v)
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
        
        // Allocate texture storage
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, g_tex_W, g_tex_H, 0, gl.RGB, gl.UNSIGNED_BYTE, null);
        
        textures.push(texture);
    }
    
    // Create Pixel Buffer Objects (PBOs) for async texture upload
    const pboSize = g_tex_W * g_tex_H * 3;  // RGB bytes
    
    for (let i = 0; i < BUFFER_SIZE; i++) {
        const pbo = gl.createBuffer();
        gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, pbo);
        gl.bufferData(gl.PIXEL_UNPACK_BUFFER, pboSize, gl.STREAM_DRAW);
        pbos.push(pbo);
    }
    
    gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindVertexArray(null);
    
    console.log('✓ Created PBOs for async texture upload');
    
    return true;
}

// Blocking queue: put() - called by worker message handler
function queuePut(item) {
    imageQueue.push(item);
    
    // If someone is waiting for an item, give it to them
    if (queueWaiters.length > 0) {
        const resolve = queueWaiters.shift();
        const nextItem = imageQueue.shift();
        resolve(nextItem);
    }
    
    // Notify worker if queue now has space
    if (imageQueue.length < MAX_QUEUE_SIZE) {
        worker.postMessage({ type: 'queue_has_space' });
    }
}

// Blocking queue: get() - blocks until item is available
function queueGet() {
    return new Promise((resolve) => {
        if (imageQueue.length > 0) {
            // Item available immediately
            resolve(imageQueue.shift());
            
            // Notify worker that queue has space
            if (imageQueue.length < MAX_QUEUE_SIZE) {
                worker.postMessage({ type: 'queue_has_space' });
            }
        } else {
            // Wait for item
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

// Upload entire texture at once (for initial buffer fill)
function uploadTextureComplete(slotIdx, imageData) {
    const rgbData = convertRGBAtoRGB(imageData);
    
    gl.bindTexture(gl.TEXTURE_2D, textures[slotIdx]);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, g_tex_W, g_tex_H, gl.RGB, gl.UNSIGNED_BYTE, rgbData);
    gl.bindTexture(gl.TEXTURE_2D, null);
}

// Upload a chunk of texture using PBO (async)
function uploadTextureChunkWithPBO(texIdx, pboIdx, rgbData, yOffset, chunkHeight) {
    const rowBytes = g_tex_W * 3;
    const startByte = yOffset * rowBytes;
    const chunkBytes = chunkHeight * rowBytes;
    const chunkData = rgbData.subarray(startByte, startByte + chunkBytes);
    
    // Bind PBO and upload data to it
    gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, pbos[pboIdx]);
    gl.bufferSubData(gl.PIXEL_UNPACK_BUFFER, startByte, chunkData);
    
    // Upload from PBO to texture (async on GPU)
    gl.bindTexture(gl.TEXTURE_2D, textures[texIdx]);
    gl.texSubImage2D(
        gl.TEXTURE_2D, 0,
        0, yOffset,              // x_offset, y_offset
        g_tex_W, chunkHeight,    // width, height of chunk
        gl.RGB, gl.UNSIGNED_BYTE,
        startByte                // offset into PBO
    );
    
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, null);
}

// Process progressive upload if one is pending
function processProgressiveUpload() {
    if (!pendingUpload) return;
    
    const { slotIdx, pboIdx, rgbData, rowsUploaded } = pendingUpload;
    
    // Calculate how much to upload this frame
    const rowsLeft = g_tex_H - rowsUploaded;
    const framesLeft = Math.max(1, Math.floor((1.0 - t) / dt));
    const rowsToUpload = Math.min(rowsLeft, Math.floor(rowsLeft / framesLeft) + 1);
    
    // Upload the chunk using PBO
    uploadTextureChunkWithPBO(slotIdx, pboIdx, rgbData, rowsUploaded, rowsToUpload);
    
    // Update state
    pendingUpload.rowsUploaded += rowsToUpload;
    
    // Check if upload is complete
    if (pendingUpload.rowsUploaded >= g_tex_H) {
        pendingUpload = null;
    }
}

// Render the scene
function render() {
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    
    gl.useProgram(program);
    
    // Bind texA (current texture)
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, textures[currTexIdx]);
    const texALoc = gl.getUniformLocation(program, 'texA');
    gl.uniform1i(texALoc, 0);
    
    // Bind texB (next texture)
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, textures[nextTexIdx]);
    const texBLoc = gl.getUniformLocation(program, 'texB');
    gl.uniform1i(texBLoc, 1);
    
    // Set interpolation parameter
    const tLoc = gl.getUniformLocation(program, 't');
    gl.uniform1f(tLoc, t);
    
    // Draw quad
    gl.bindVertexArray(vao);
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_INT, 0);
    
    gl.bindVertexArray(null);
}

// Update FPS counter and stats
function updateFPS() {
    frameCount++;
    const now = performance.now();
    const elapsed = now - lastFpsUpdate;
    
    // Frame time
    const currentFrameTime = now - lastFrameTime;
    frameTimeAvg = frameTimeAvg * 0.9 + currentFrameTime * 0.1;
    lastFrameTime = now;
    
    if (elapsed >= 1000) {  // Update every second
        fps = Math.round((frameCount * 1000) / elapsed);
        document.getElementById('fps').textContent = 
            `FPS: ${fps} | Frame: ${frameTimeAvg.toFixed(1)}ms | Worker: ${workerTimeAvg.toFixed(1)}ms | Queue: ${imageQueue.length}/${MAX_QUEUE_SIZE}`;
        frameCount = 0;
        lastFpsUpdate = now;
    }
}

// Animation loop
async function animate() {
    // STEP 1: Process progressive upload if pending (matches Python)
    if (pendingUpload) {
        processProgressiveUpload();
    }
    
    // STEP 2: Check if we need to advance to next texture pair (matches Python - BEFORE render)
    if (t > 1.0) {
        t = 0.0;
        currTexIdx = (currTexIdx + 1) % BUFFER_SIZE;
        nextTexIdx = (nextTexIdx + 1) % BUFFER_SIZE;
        
        // Assert no pending upload (should have finished by now)
        if (pendingUpload !== null) {
            console.error('❌ ERROR: Pending upload not finished!');
        }
        
        // BLOCKING GET from queue
        const { slotIdx, imageData, generationTime } = await queueGet();
        
        // Update worker time average
        workerTimeAvg = workerTimeAvg * 0.9 + generationTime * 0.1;
        
        const rgbData = convertRGBAtoRGB(new Uint8ClampedArray(imageData));
        
        // Start progressive upload using corresponding PBO
        pendingUpload = {
            slotIdx,
            pboIdx: slotIdx,
            rgbData,
            rowsUploaded: 0
        };
    }
    
    // STEP 3: Render current frame (matches Python)
    render();
    updateFPS();
    
    // STEP 4: Advance interpolation (matches Python)
    t += dt;
    
    requestAnimationFrame(animate);
}

// Initialize worker
function initWorker() {
    worker = new Worker('worker.js');
    
    worker.onmessage = function(e) {
        const { type, slotIdx, data, generationTime } = e.data;
        
        switch(type) {
            case 'ready':
                isReady = true;
                console.log('Worker ready!');
                // Start the worker generating
                worker.postMessage({ type: 'start_generating' });
                // Wait for buffer to fill
                fillBuffer();
                break;
                
            case 'image':
                // BLOCKING PUT to queue
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
        alert('Worker failed to initialize');
    };
}

// Fill buffer before starting animation
async function fillBuffer() {
    console.log('Filling buffer...');
    
    // BLOCKING: Get BUFFER_SIZE images
    for (let i = 0; i < BUFFER_SIZE; i++) {
        const { slotIdx, imageData, generationTime } = await queueGet();
        uploadTextureComplete(slotIdx, new Uint8ClampedArray(imageData));
        console.log(`Buffer filled: ${i + 1}/${BUFFER_SIZE} (slot ${slotIdx}, generated in ${generationTime.toFixed(1)}ms)`);
    }
    
    console.log('Buffer full, starting animation...');
    animate();  // Start animation loop
}

// Initialize on load
if (initWebGL()) {
    initWorker();
} else {
    alert('Failed to initialize WebGL');
}