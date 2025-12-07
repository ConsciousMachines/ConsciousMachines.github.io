// Import TensorFlow.js
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.11.0');

// --- GLOBALS
const BUFFER_SIZE = 3;
const NUM_PCS = 84;
const HP = 128;
const WP = 128;
const HG = 4;
const WG = 6;
const g_tex_H = HP * HG;
const g_tex_W = WP * WG;

// PCA control parameters
const k = 0.9;
const s = 0.9;
const p = 1.0;
const num_pc = NUM_PCS;

// File paths
const stds_file = 'data/stds.bin';
const mu_file = 'data/mu.bin';
const eigvecs_file = 'data/eigvecs.bin';

let stds, mu, eigvecs, z;
let slotIdx = 0;
let isInitialized = false;

// Queue control - signals when we can put into queue
let queueSpaceAvailable = 0;
let queueSpaceWaiters = [];

// Load binary file as Float32Array
async function loadBinaryFile(url) {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    return new Float32Array(buffer);
}

// Wait for queue to have space (blocking put)
function waitToSend() {
    return new Promise((resolve) => {
        if (queueSpaceAvailable > 0) {
            queueSpaceAvailable--;
            resolve();
        } else {
            queueSpaceWaiters.push(resolve);
        }
    });
}

// Initialize TensorFlow and data
async function init() {
    console.log('Initializing TensorFlow.js...');
    
    try {
        // Try to use WebGL with OffscreenCanvas
        if (typeof OffscreenCanvas !== 'undefined') {
            console.log('OffscreenCanvas available, trying WebGL backend...');
            
            const canvas = new OffscreenCanvas(1, 1);
            const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
            
            if (gl) {
                console.log('WebGL context created successfully');
                await tf.setBackend('webgl');
                await tf.ready();
                console.log('TensorFlow.js ready with backend:', tf.getBackend());
            } else {
                console.log('WebGL context failed, falling back to CPU');
                await tf.setBackend('cpu');
                await tf.ready();
                console.log('TensorFlow.js ready with backend:', tf.getBackend());
            }
        } else {
            console.log('OffscreenCanvas not available, using CPU backend');
            await tf.setBackend('cpu');
            await tf.ready();
            console.log('TensorFlow.js ready with backend:', tf.getBackend());
        }
        
        console.log('Loading data files...');
        
        const [stds_data, mu_data, eigvecs_data] = await Promise.all([
            loadBinaryFile(stds_file),
            loadBinaryFile(mu_file),
            loadBinaryFile(eigvecs_file)
        ]);

        stds = tf.tensor1d(stds_data);
        mu = tf.tensor1d(mu_data);
        eigvecs = tf.tensor2d(eigvecs_data, [NUM_PCS, HP * WP * 3]);

        // Initialize z for AR(1) process
        z = tf.mul(tf.randomNormal([HG * WG, NUM_PCS]), stds);

        isInitialized = true;
        console.log('Data loaded successfully!');
        console.log('Memory info:', tf.memory());
        
        self.postMessage({ type: 'ready' });
    } catch (err) {
        console.error('Initialization error:', err);
        self.postMessage({ type: 'error', data: err.message });
    }
}

// Generate single image
async function generateImage() {
    if (!isInitialized) {
        console.error('Cannot generate image: not initialized');
        return null;
    }
    
    const startTime = performance.now();
    
    try {
        const result = tf.tidy(() => {
            const indices = tf.range(0, NUM_PCS, 1, 'float32');

            // Update z with AR(1) process
            let mask = tf.cast(tf.less(tf.randomUniform([HG * WG, NUM_PCS]), p), 'float32');
            mask = tf.mul(mask, tf.cast(tf.less(indices, num_pc), 'float32'));
            
            const noise = tf.randomNormal([HG * WG, NUM_PCS]);
            const update = tf.mul(
                tf.mul(tf.sqrt(1 - k * k), s),
                tf.mul(tf.mul(mask, noise), stds)
            );
            
            // Store old z to dispose later
            const oldZ = z;
            z = tf.add(tf.mul(k, z), update);
            tf.keep(z);
            oldZ.dispose();

            // Generate image
            const z_slice = tf.slice(z, [0, 0], [HG * WG, num_pc]);
            const eigvecs_slice = tf.slice(eigvecs, [0, 0], [num_pc, HP * WP * 3]);
            
            let x = tf.add(mu, tf.matMul(z_slice, eigvecs_slice));
            x = tf.reshape(x, [HG, WG, HP, WP, 3]);
            x = tf.transpose(x, [0, 2, 1, 3, 4]);
            x = tf.reshape(x, [HG * HP, WG * WP, 3]);
            x = tf.clipByValue(x, 0.0, 255.0);
            x = tf.cast(x, 'int32');

            return x;
        });
        
        // Extract image data
        const data = await result.data();
        result.dispose();
        
        // Convert to RGBA format
        const rgbaData = new Uint8ClampedArray(HG * HP * WG * WP * 4);
        for (let i = 0; i < HG * HP * WG * WP; i++) {
            rgbaData[i * 4] = data[i * 3];
            rgbaData[i * 4 + 1] = data[i * 3 + 1];
            rgbaData[i * 4 + 2] = data[i * 3 + 2];
            rgbaData[i * 4 + 3] = 255;
        }
        
        const generationTime = performance.now() - startTime;
        
        const currentSlot = slotIdx;
        slotIdx = (slotIdx + 1) % BUFFER_SIZE;
        
        return { slotIdx: currentSlot, data: rgbaData.buffer, generationTime };
    } catch (err) {
        console.error('Generation error:', err);
        self.postMessage({ type: 'error', data: err.message });
        return null;
    }
}

// Continuously generate images (matches Python pattern)
async function continuousGenerate() {
    while (true) {
        // STEP 1: Generate image (NOT blocking)
        const result = await generateImage();
        
        if (result) {
            // STEP 2: Wait for queue space (BLOCKING)
            await waitToSend();
            
            // STEP 3: Send to main thread
            self.postMessage({
                type: 'image',
                slotIdx: result.slotIdx,
                data: result.data,
                generationTime: result.generationTime
            }, [result.data]);
        }
    }
}

// Message handler
self.onmessage = function(e) {
    const { type } = e.data;
    
    switch(type) {
        case 'start_generating':
            // Initial signal - queue has BUFFER_SIZE spaces
            queueSpaceAvailable = BUFFER_SIZE;
            continuousGenerate();
            break;
            
        case 'queue_has_space':
            // Main thread signals that queue has space (after consuming an item)
            if (queueSpaceWaiters.length > 0) {
                const resolve = queueSpaceWaiters.shift();
                resolve();
            } else {
                queueSpaceAvailable++;
            }
            break;
    }
};

// Initialize on load
init();