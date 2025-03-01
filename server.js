// server.js - Main server file that will run on Northflank

const express = require('express');
const http = require('http');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const WebSocket = require('ws');

// Initialize Express app
const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));

// WebSocket connection for real-time streaming
wss.on('connection', (ws) => {
  console.log('Client connected');
  
  ws.on('message', (message) => {
    try {
      const data = JSON.parse(message);
      
      if (data.type === 'simulation_params') {
        // Start or update the simulation with the received parameters
        runSimulation(data.params, ws);
      }
    } catch (error) {
      console.error('Error processing message:', error);
    }
  });
  
  ws.on('close', () => {
    console.log('Client disconnected');
    // Clean up any GPU resources for this client
  });
});

// Function to run the ocean simulation on a GPU
function runSimulation(params, ws) {
  // This is where we'll integrate with a GPU processing library 
  // or spawn a process that uses CUDA/OpenCL for H100s
  
  // For example, using TensorFlow.js with GPU support:
  // const tf = require('@tensorflow/tfjs-node-gpu');
  
  // Or spawning a Python process that uses libraries with GPU support
  const pythonProcess = spawn('python', [
    path.join(__dirname, 'gpu_simulation', 'run_simulation.py'),
    JSON.stringify(params)
  ]);
  
  pythonProcess.stdout.on('data', (data) => {
    // Parse the data (could be frame data, metrics, etc.)
    const simulationOutput = data.toString();
    
    // Send it to the client
    ws.send(JSON.stringify({
      type: 'simulation_frame',
      data: simulationOutput
    }));
  });
  
  pythonProcess.stderr.on('data', (data) => {
    console.error(`Simulation error: ${data}`);
    ws.send(JSON.stringify({
      type: 'error',
      message: data.toString()
    }));
  });
}

// REST API routes
app.get('/api/status', (req, res) => {
  // Return status of GPU servers
  res.json({ status: 'running', gpus: 8, utilization: '75%' });
});

app.post('/api/simulation/start', express.json(), (req, res) => {
  // Handle simulation start request
  // This could be used for one-off simulations rather than real-time
  const params = req.body;
  
  // Start a new simulation job
  // ...
  
  res.json({ status: 'started', jobId: 'sim_123456' });
});

// Start the server
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});