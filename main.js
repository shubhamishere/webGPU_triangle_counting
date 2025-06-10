// This script orchestrates the entire triangle counting process.

// Get references to all the HTML elements we need to interact with.
const graphInput = document.getElementById('graphInput');
const runButton = document.getElementById('runButton');
const triangleCountEl = document.getElementById('triangleCount');
const statusEl = document.getElementById('status');

// Disable the run button initially. It will be enabled when a file is selected.
runButton.disabled = true;

// Add an event listener to the file input to enable the run button.
graphInput.addEventListener('change', () => {
    if (graphInput.files.length > 0) {
        runButton.disabled = false;
        runButton.textContent = 'Run Triangle Counting';
    } else {
        runButton.disabled = true;
        runButton.textContent = 'Select a file first';
    }
});

// Add a click event listener to the run button to start the process.
runButton.addEventListener('click', async () => {
    if (graphInput.files.length === 0) {
        updateStatus('Please select a graph file first.', 'error');
        return;
    }
    // Reset count before running
    triangleCountEl.textContent = '0';
    await runTriangleCounting();
});

/**
 * Updates the status message and its appearance on the UI.
 * @param {string} message The text to display.
 * @param {'idle'|'running'|'success'|'error'} type The type of status.
 */
function updateStatus(message, type = 'running') {
    statusEl.textContent = message;
    statusEl.classList.remove('text-yellow-400', 'text-green-400', 'text-red-400', 'animate-pulse');
    switch (type) {
        case 'success':
            statusEl.classList.add('text-green-400');
            break;
        case 'error':
            statusEl.classList.add('text-red-400');
            break;
        case 'running':
            statusEl.classList.add('text-yellow-400', 'animate-pulse');
            break;
        default: // idle
            statusEl.classList.add('text-yellow-400');
    }
}


/**
 * Parses the user-provided graph file and converts it to CSR format.
 * This version is more robust and includes logging for debugging.
 * @param {File} file The .txt file selected by the user.
 * @returns {Promise<{numNodes: number, rowPtr: Uint32Array, edgeList: Uint32Array}>} The graph in CSR format.
 */
async function parseGraphToCSR(file) {
    console.log('--- Starting Graph Parsing ---');
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const text = event.target.result;
                const lines = text.split('\n').filter(line => line.trim() !== '' && !line.startsWith('#'));
                
                const adj = new Map();
                const allNodeIds = new Set();

                lines.forEach(line => {
                    const [u_str, v_str] = line.trim().split(/\s+/);
                    const u = parseInt(u_str, 10);
                    const v = parseInt(v_str, 10);

                    if (isNaN(u) || isNaN(v)) return;

                    allNodeIds.add(u);
                    allNodeIds.add(v);
                    
                    if (!adj.has(u)) adj.set(u, new Set());
                    if (!adj.has(v)) adj.set(v, new Set());
                    
                    if (u !== v) {
                       adj.get(u).add(v);
                       adj.get(v).add(u);
                    }
                });

                // Relabel nodes to a contiguous 0..N-1 range.
                const sortedNodeIds = Array.from(allNodeIds).sort((a, b) => a - b);
                const nodeMap = new Map(sortedNodeIds.map((id, index) => [id, index]));
                
                const numNodes = sortedNodeIds.length;
                console.log(`Discovered ${numNodes} unique nodes.`);

                const rowPtr = new Uint32Array(numNodes + 1);
                const edgeListData = [];

                let currentEdgeIndex = 0;
                for (let i = 0; i < numNodes; i++) {
                    rowPtr[i] = currentEdgeIndex;
                    const originalId = sortedNodeIds[i];

                    if (adj.has(originalId)) {
                        const neighbors = Array.from(adj.get(originalId))
                                           .sort((a, b) => a - b) // Sort neighbors
                                           .map(neighborId => nodeMap.get(neighborId)); // Remap neighbors

                        for (const neighbor of neighbors) {
                            edgeListData.push(neighbor);
                        }
                        currentEdgeIndex += neighbors.length;
                    }
                }
                rowPtr[numNodes] = currentEdgeIndex;

                const edgeList = new Uint32Array(edgeListData);
                
                // --- DEBUGGING LOGS ---
                console.log("Graph parsing complete. Data to be sent to GPU:");
                console.log("Number of Nodes:", numNodes);
                console.log("Row Pointers (CSR Offset Array):", rowPtr);
                console.log("Edge List (CSR Column Index Array):", edgeList);
                console.log('--- End of Graph Parsing ---');

                resolve({ numNodes, rowPtr, edgeList });
            } catch (e) {
                console.error("Error during graph parsing:", e);
                reject(e);
            }
        };
        reader.onerror = (error) => reject(error);
        reader.readAsText(file);
    });
}

/**
 * Main function to set up WebGPU, run the computation, and display results.
 */
async function runTriangleCounting() {
    try {
        // 1. Initialize WebGPU
        updateStatus('Initializing WebGPU...', 'running');
        if (!navigator.gpu) {
            updateStatus('WebGPU not supported on this browser.', 'error');
            throw new Error('WebGPU not supported.');
        }
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            updateStatus('Failed to get GPU adapter.', 'error');
            throw new Error('No appropriate GPUAdapter found.');
        }
        const device = await adapter.requestDevice();
        updateStatus('WebGPU Initialized.', 'success');

        // 2. Parse Graph Data
        updateStatus('Parsing graph file...', 'running');
        const { numNodes, rowPtr, edgeList } = await parseGraphToCSR(graphInput.files[0]);
        updateStatus(`Graph parsed: ${numNodes} nodes, ${edgeList.length / 2} edges.`, 'success');

        // 3. Create GPU Buffers and copy data
        updateStatus('Uploading graph data to GPU...', 'running');
        
        const rowPtrBuffer = device.createBuffer({
            size: rowPtr.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint32Array(rowPtrBuffer.getMappedRange()).set(rowPtr);
        rowPtrBuffer.unmap();
        
        const edgeListBuffer = device.createBuffer({
            size: edgeList.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint32Array(edgeListBuffer.getMappedRange()).set(edgeList);
        edgeListBuffer.unmap();

        const resultBuffer = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        const stagingBuffer = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        updateStatus('Data uploaded to GPU.', 'success');

        // 4. Create Shader Module and Compute Pipeline
        updateStatus('Compiling GPU shader...', 'running');
        const shaderCode = await fetch('compute.wgsl').then(res => res.text());
        const shaderModule = device.createShaderModule({ code: shaderCode });

        const computePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });

        // 5. Create Bind Group
        const bindGroup = device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: rowPtrBuffer } },
                { binding: 1, resource: { buffer: edgeListBuffer } },
                { binding: 2, resource: { buffer: resultBuffer } },
            ],
        });

        // 6. Dispatch Computation
        updateStatus('Running computation on GPU...', 'running');
        const commandEncoder = device.createCommandEncoder();
        
        commandEncoder.clearBuffer(resultBuffer);

        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        
        const workgroupSize = 256;
        const numWorkgroups = Math.ceil(numNodes / workgroupSize);
        passEncoder.dispatchWorkgroups(numWorkgroups);
        
        passEncoder.end();

        commandEncoder.copyBufferToBuffer(resultBuffer, 0, stagingBuffer, 0, 4);

        device.queue.submit([commandEncoder.finish()]);

        // 7. Read Result Back
        // *** FIX: Wait for the GPU to completely finish its work, not just for submission. ***
        await device.queue.onSubmittedWorkDone(); 
        updateStatus('Awaiting GPU result...', 'running');

        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const resultArray = new Uint32Array(stagingBuffer.getMappedRange());
        const count = resultArray[0];
        stagingBuffer.unmap();

        const finalCount = count; 

        // 8. Display Result and Clean Up
        triangleCountEl.textContent = finalCount.toLocaleString();
        updateStatus('Computation successful!', 'success');
        
        rowPtrBuffer.destroy();
        edgeListBuffer.destroy();
        resultBuffer.destroy();
        stagingBuffer.destroy();
    } catch (error) {
        console.error("An error occurred in runTriangleCounting:", error);
        updateStatus(`Error: ${error.message}`, 'error');
    }
}
