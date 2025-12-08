// Import TensorFlow.js
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.11.0');

// --- GLOBALS
const BUFFER_SIZE = 3;
const NUM_PCS = 84;
const HP = 128;
const WP = 128;

// Dynamic grid size (set by main thread)
let HG = 4;
let WG = 6;
let g_tex_H = HP * HG;
let g_tex_W = WP * WG;

// // PCA control
// const k = 0.9;
// const s = 0.9;
// const p = 1.0;
// const num_pc = NUM_PCS;

// File paths
const stds_file = 'data/stds.bin';
const mu_file = 'data/mu.bin';
const eigvecs_file = 'data/eigvecs.bin';

let stds, mu, eigvecs, z;
let slotIdx = 0;
let isInitialized = false;
let queueSpaceAvailable = 0;
let queueSpaceWaiters = [];

// PCA control (now mutable)
let k = 0.9;
let s = 0.9;
let p = 1.0;
let num_pc = NUM_PCS;

async function loadBinaryFile(url) {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    return new Float32Array(buffer);
}

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

async function init() {
    console.log('Initializing TensorFlow.js...');
    
    try {
        if (typeof OffscreenCanvas !== 'undefined') {
            const canvas = new OffscreenCanvas(1, 1);
            const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
            
            if (gl) {
                await tf.setBackend('webgl');
                await tf.ready();
                console.log('TensorFlow.js ready with backend: webgl');
            } else {
                await tf.setBackend('cpu');
                await tf.ready();
                console.log('TensorFlow.js ready with backend: cpu');
            }
        } else {
            await tf.setBackend('cpu');
            await tf.ready();
            console.log('TensorFlow.js ready with backend: cpu');
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
        console.log(`Data loaded successfully! Grid: ${HG}x${WG}`);
        
        self.postMessage({ type: 'ready' });
    } catch (err) {
        console.error('Initialization error:', err);
        self.postMessage({ type: 'error', data: err.message });
    }
}

async function generateImage() {
    if (!isInitialized) return null;
    
    const startTime = performance.now();
    
    try {
        const result = tf.tidy(() => {
            const indices = tf.range(0, NUM_PCS, 1, 'float32');

            let mask = tf.cast(tf.less(tf.randomUniform([HG * WG, NUM_PCS]), p), 'float32');
            mask = tf.mul(mask, tf.cast(tf.less(indices, num_pc), 'float32'));
            
            const noise = tf.randomNormal([HG * WG, NUM_PCS]);
            const update = tf.mul(
                tf.mul(tf.sqrt(1 - k * k), s),
                tf.mul(tf.mul(mask, noise), stds)
            );
            
            const oldZ = z;
            z = tf.add(tf.mul(k, z), update);
            tf.keep(z);
            oldZ.dispose();

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
        
        const data = await result.data();
        result.dispose();
        
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

async function continuousGenerate() {
    while (true) {
        const result = await generateImage();
        
        if (result) {
            await waitToSend();
            
            self.postMessage({
                type: 'image',
                slotIdx: result.slotIdx,
                data: result.data,
                generationTime: result.generationTime
            }, [result.data]);
        }
    }
}

self.onmessage = function(e) {
    const { type } = e.data;
    
    switch(type) {
        case 'set_grid':
            // Update grid dimensions
            HG = e.data.HG;
            WG = e.data.WG;
            g_tex_H = HP * HG;
            g_tex_W = WP * WG;
            console.log(`Worker grid set to ${HG}x${WG}`);
            
            // Reinitialize z with new dimensions if already initialized
            if (isInitialized && z) {
                z.dispose();
                z = tf.mul(tf.randomNormal([HG * WG, NUM_PCS]), stds);
                tf.keep(z);
            }
            
            init();
            break;

        case 'update_params':
            // Update PCA parameters
            k = e.data.params.k;
            s = e.data.params.s;
            p = e.data.params.p;
            num_pc = e.data.params.num_pc;
            console.log(`Params updated: k=${k}, s=${s}, p=${p}, num_pc=${num_pc}`);
            break;
            
        case 'start_generating':
            queueSpaceAvailable = BUFFER_SIZE;
            continuousGenerate();
            break;
            
        case 'queue_has_space':
            if (queueSpaceWaiters.length > 0) {
                const resolve = queueSpaceWaiters.shift();
                resolve();
            } else {
                queueSpaceAvailable++;
            }
            break;
    }
};