FROM node:18

WORKDIR /app

# Create a simple test server file directly in the Dockerfile
RUN echo 'console.log("Starting test server"); \
          const http = require("http"); \
          const server = http.createServer((req, res) => { \
            res.writeHead(200); \
            res.end("Test server running"); \
          }); \
          server.listen(3000, () => console.log("Test server running on port 3000"));' > /app/server.js

# Display files for debugging
RUN ls -la /app

EXPOSE 3000

CMD ["node", "server.js"]