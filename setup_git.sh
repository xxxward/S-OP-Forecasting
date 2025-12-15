#!/bin/bash

# Git Setup Script for Demand Forecast App

echo "🚀 Setting up Git repository for Demand Forecast App"
echo ""

# Initialize git repository
echo "📦 Initializing git repository..."
git init

# Add all files
echo "📝 Adding files to git..."
git add .

# Create initial commit
echo "💾 Creating initial commit..."
git commit -m "Initial commit: Demand forecast app with Google Sheets integration

- Added Streamlit app with Google Sheets connection
- Added requirements and dependencies
- Added devcontainer configuration for Codespaces
- Added utility functions for data processing
- Added comprehensive documentation"

echo ""
echo "✅ Git repository initialized!"
echo ""
echo "📋 Next steps:"
echo "1. Create a new repository on GitHub"
echo "2. Run: git remote add origin <your-repo-url>"
echo "3. Run: git branch -M main"
echo "4. Run: git push -u origin main"
echo ""
echo "🎉 Your repository is ready to push!"
