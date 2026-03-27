# Deploying to Render

This guide explains how to deploy your AI-Powered SDSS project to Render.

## Prerequisites

1. **GitHub Repository**: Push your project to a GitHub repository
2. **Render Account**: Create a free account at [render.com](https://render.com)
3. **API Keys**: 
   - OpenAI API key (for AI insights)
   - Optional: Groq API key (alternative LLM provider)

## Deployment Steps

### Option 1: Using GitHub Integration (Recommended)

1. **Go to Render Dashboard**
   - Visit [https://dashboard.render.com](https://dashboard.render.com)
   - Click "New +" and select "Web Service"

2. **Connect Your Repository**
   - Select "Deploy from a Git repository"
   - Authorize GitHub and select your repository
   - Select the branch (usually "main")

3. **Configure Service**
   - **Name**: ai-powered-sdss (or your preferred name)
   - **Environment**: Python 3
   - **Region**: Choose closest to your users
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8080`

4. **Set Environment Variables**
   In the "Environment" section, add:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `GROQ_API_KEY`: (Optional) Your Groq API key
   - `APP_HOST`: `0.0.0.0`
   - `APP_PORT`: `8080`
   - `ALLOWED_ORIGINS`: `*` (or your specific domains)

5. **Deploy**
   - Click "Create Web Service"
   - Render will automatically deploy your application
   - Wait for the build to complete (usually 2-5 minutes)

### Option 2: Using render.yaml

1. **Commit Configuration**
   - The `render.yaml` file in your repository defines the service
   - Commit this file to your repository

2. **Deploy**
   - Go to Render Dashboard
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Render will auto-detect and deploy based on render.yaml

## Environment Variables for Render

Set these in the Render dashboard:

| Variable | Value | Description |
|----------|-------|-------------|
| `OPENAI_API_KEY` | Your key | Required for AI features |
| `GROQ_API_KEY` | Your key | Optional alternative LLM |
| `APP_HOST` | `0.0.0.0` | Bind to all interfaces |
| `APP_PORT` | `8080` | Port (auto-assigned by Render) |
| `ALLOWED_ORIGINS` | `*` | CORS origins (adjust for security) |

## Local Development

For local testing before deployment:

1. **Copy environment file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit .env** with your local values:
   ```
   APP_HOST=localhost
   APP_PORT=8001
   OPENAI_API_KEY=sk-your-key
   ```

3. **Run locally**:
   ```bash
   # Windows PowerShell
   .\start_backend.ps1

   # Or manually
   cd backend
   python -m uvicorn main:app --host localhost --port 8001
   ```

4. **Access the application**:
   - Open browser to `http://localhost:8001`

## Troubleshooting

### "Module not found" errors
- Check that `requirements.txt` is in the `backend/` directory
- Verify all imports are listed in requirements.txt

### Port/Host errors
- Render automatically assigns a port; don't hardcode port 8001
- Use environment variables: `APP_PORT` and `APP_HOST`
- The app is correctly configured now

### Frontend not loading
- The backend now serves static frontend files
- No separate frontend deployment needed
- All files are served from one endpoint

### CORS errors in browser
- Check `ALLOWED_ORIGINS` environment variable
- In development: include your local URL
- In production: include your domain

### API calls return 404
- Verify the API URL in frontend JavaScript is using `getAPIUrl()` function
- Check that backend is correctly serving static files
- Verify all routes in `main.py` are properly defined

## File Structure

```
AI-Powered-SDSS/
├── backend/
│   ├── main.py                 # FastAPI backend (now serves frontend)
│   ├── requirements.txt         # Python dependencies
│   ├── ai_advisor.py
│   ├── rag_chatbot.py
│   └── what_if_simulator.py
├── frontend/
│   ├── index.html             # Frontend SPA
│   ├── script.js              # Updated with dynamic API URL
│   ├── cleaning.js            # Updated with dynamic API URL
│   ├── ml-training.js         # Updated with dynamic API URL
│   ├── dataset.js             # Updated with dynamic API URL
│   └── [other frontend files]
├── Procfile                    # Render deployment config
├── render.yaml                 # Render blueprint configuration
├── .env.example                # Environment variables template
└── README.md
```

## What Changed for Render

1. **Backend (main.py)**:
   - Added static file serving for frontend
   - Environment variables for host/port configuration
   - Flexible CORS configuration
   - Routes to serve HTML files and SPA routing

2. **Frontend (JavaScript)**:
   - Replaced hardcoded `API_URL` with `getAPIUrl()` function
   - Dynamically detects environment (localhost vs production)
   - Works in both local development and cloud deployment

3. **Dependencies**:
   - Added `python-dotenv` for environment variable support
   - Added `groq` for optional Groq API support

4. **Configuration**:
   - Created `Procfile` for Heroku-compatible deployment
   - Created `render.yaml` for Render blueprint deployment
   - Created `.env.example` for environment variable reference

## Deployment Status Check

1. **Check Render Logs**:
   - Go to your service on Render dashboard
   - Click on "Logs" tab to see deployment and runtime logs

2. **Test the Application**:
   - Open the service URL (provided by Render)
   - Upload a dataset and test the workflow
   - Check browser console for any errors

3. **Monitor Performance**:
   - Render provides analytics on the service dashboard
   - Monitor CPU, memory, and request times

## Next Steps

After successful deployment:

1. **Configure Custom Domain** (optional):
   - In Render dashboard, add custom domain settings
   - Point your domain to Render

2. **Set up Auto-Deployment**:
   - Render auto-deploys on git push to selected branch
   - Verify webhook is properly configured

3. **Production Considerations**:
   - Use strong API keys (not development keys)
   - Limit `ALLOWED_ORIGINS` to your domain
   - Monitor logs for errors
   - Set up backup procedures for uploaded datasets
   - Consider file storage solution for uploaded files

## Support

For issues specific to Render:
- [Render Documentation](https://render.com/docs)
- [Render Support](https://support.render.com)

For project-specific issues:
- Check the README.md in project root
- Review deployment logs on Render dashboard
- Test locally first before deploying to Render
