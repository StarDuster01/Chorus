#!/bin/bash
echo "Starting RAGBot Application..."
echo ""
echo "If this is your first time running, make sure to configure the .env file!"
echo ""
echo "Press Ctrl+C to stop the application."
echo ""
xdg-open http://localhost:5000 &> /dev/null || open http://localhost:5000 &> /dev/null &
./RAGBot