# Security Configuration

## Environment Variables Setup

E-Raksha uses environment variables to keep sensitive information secure. **Never commit API keys or secrets to the repository.**

### Required Setup

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your actual credentials:**
   ```bash
   # Supabase Configuration (Optional - for database features)
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_ANON_KEY=your_anon_key_here
   
   # API Configuration
   API_HOST=0.0.0.0
   API_PORT=8000
   FRONTEND_PORT=3001
   ```

### What's Protected

- **Supabase credentials** - Database access keys
- **API keys** - External service authentication
- **Secret keys** - Application security tokens
- **Database URLs** - Connection strings
- **Kaggle credentials** - Model training access

### Docker Deployment

The Docker setup automatically loads environment variables from `.env` file:

```bash
# Create your .env file first
cp .env.example .env
# Edit .env with your credentials

# Then deploy
docker-compose up --build
```

### Production Deployment

For production deployments, set environment variables in your hosting platform:

#### Railway
```bash
# Set via Railway dashboard or CLI
railway variables set SUPABASE_URL=your_url
railway variables set SUPABASE_ANON_KEY=your_key
```

#### Render
```bash
# Set in Render dashboard under Environment Variables
SUPABASE_URL=your_url
SUPABASE_ANON_KEY=your_key
```

#### AWS/Docker
```bash
# Set via environment variables in container
docker run -e SUPABASE_URL=your_url -e SUPABASE_ANON_KEY=your_key eraksha:latest
```

### Optional Features

**Database features are optional.** If you don't set Supabase credentials:
- Core deepfake detection still works
- Web interface functions normally  
- No inference logging to database
- No user feedback storage
- No usage statistics

### Security Best Practices

1. **Never commit `.env` files** - They're in `.gitignore`
2. **Use different keys for development/production**
3. **Rotate keys regularly**
4. **Use least-privilege access** - Only grant necessary permissions
5. **Monitor access logs** - Check for unauthorized usage

### Troubleshooting

**"Supabase credentials not found"**
- This is normal if you haven't set up database features
- Core functionality works without database
- Set `SUPABASE_URL` and `SUPABASE_ANON_KEY` to enable database features

**"Environment variables not loading"**
- Ensure `.env` file exists in project root
- Check file permissions (should be readable)
- Verify no extra spaces in variable assignments

### Getting Supabase Credentials (Optional)

1. Go to [supabase.com](https://supabase.com)
2. Create a free account and project
3. Go to Settings > API
4. Copy your Project URL and anon/public key
5. Add to your `.env` file

**Note:** Database features are optional. E-Raksha works perfectly without them!