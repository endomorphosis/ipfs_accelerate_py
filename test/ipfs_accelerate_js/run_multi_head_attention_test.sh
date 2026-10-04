#!/bin/bash

# Script to run the multi-head attention benchmark in different browsers

# Check if the browser is specified
if [ $# -eq 0 ]; then
  echo "Usage: $0 [chrome|firefox|safari|edge]"
  echo "Example: $0 chrome"
  exit 1
fi

BROWSER=$1

# Set up the browser command based on the specified browser
case $BROWSER in
  chrome)
    BROWSER_CMD="google-chrome"
    ;;
  firefox)
    BROWSER_CMD="firefox"
    ;;
  safari)
    BROWSER_CMD="safari"
    ;;
  edge)
    BROWSER_CMD="microsoft-edge"
    ;;
  *)
    echo "Unsupported browser: $BROWSER"
    echo "Supported browsers: chrome, firefox, safari, edge"
    exit 1
    ;;
esac

# Check if the browser is installed
command -v $BROWSER_CMD >/dev/null 2>&1 || { 
  echo >&2 "Browser $BROWSER_CMD is not installed. Aborting."; 
  exit 1; 
}

# Build and serve the test
echo "Building and serving the multi-head attention test..."
npm run build
npx http-server -p 8080 &
SERVER_PID=$!

# Give the server a moment to start
sleep 2

# Open the test in the specified browser
echo "Opening test in $BROWSER..."
$BROWSER_CMD "http://localhost:8080/test/multi_head_attention_test.html" &
BROWSER_PID=$!

# Wait for user to press Enter to terminate
echo "Press Enter to terminate the server and browser..."
read

# Kill the browser and server
kill $BROWSER_PID 2>/dev/null
kill $SERVER_PID 2>/dev/null

echo "Test completed."