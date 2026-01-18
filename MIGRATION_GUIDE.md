# Supabase Migration Guide

This guide will help you migrate your Hog Farm application from CSV files to Supabase database.

## Prerequisites

1. **Supabase Account**: Create a free account at [supabase.com](https://supabase.com)
2. **New Project**: Create a new project in Supabase dashboard

## Setup Steps

### 1. Create Database Tables

1. Go to your Supabase project dashboard
2. Click on "SQL Editor" in the sidebar
3. Copy and paste the contents of `supabase_schema.sql`
4. Click "Run" to execute the schema

### 2. Get Your Credentials

1. In Supabase dashboard, go to Project Settings → API
2. Copy the following:
   - **Project URL** (this is your SUPABASE_URL)
   - **anon public** key (this is your SUPABASE_KEY)
   - **service_role** key (optional, for admin operations)

### 3. Configure Environment Variables

Create a `.env` file in your project root:

```bash
# Copy from .env.example and fill in your values
cp .env.example .env
```

Edit `.env` with your credentials:

```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-key-here
SUPABASE_SERVICE_KEY=your-service-role-key-here  # Optional
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

## Migration Process

The application includes automatic migration:

1. **First Run**: The app will detect existing CSV files
2. **Migration Button**: Look for "Migrate to Supabase" option in the sidebar
3. **One-Click Migration**: Click to migrate all your data:
   - Hogs and their IDs
   - Weight measurements
   - Financial transactions
   - Budget data

## Features After Migration

✅ **Real-time Data Sync**: Changes are instantly saved to cloud  
✅ **Multi-device Access**: Access your data from any device  
✅ **Automatic Backups**: Supabase handles backups automatically  
✅ **Data Integrity**: ACID compliance prevents data corruption  
✅ **Scalability**: Handle much larger datasets efficiently  
✅ **Offline Fallback**: App still works without internet (uses CSV)  

## Troubleshooting

### Connection Issues
- Check your `.env` file has correct credentials
- Verify your Supabase project is active
- Check internet connection

### Migration Fails
- Ensure you ran the SQL schema first
- Check that your CSV files are not corrupted
- Try migrating smaller batches

### Performance Issues
- For large datasets (>10,000 records), consider:
  - Adding indexes to frequently queried columns
  - Using Supabase's built-in caching

## Data Structure

### Tables Created:
- `hogs` - Basic hog information
- `weight_measurements` - All weight data
- `financial_transactions` - Sales and expenses
- `budgets` - Budget planning
- `audit_trail` - Activity logging

### Security Features:
- Row Level Security (RLS) enabled
- Users can only access their own data
- API key authentication

## Deployment

### Streamlit Cloud
1. Add your environment variables in Streamlit Cloud dashboard
2. Deploy as usual - Supabase works perfectly in cloud

### Local Development
1. Keep your `.env` file secure (never commit to git)
2. Use different keys for development/production

## Support

- **Supabase Docs**: [supabase.com/docs](https://supabase.com/docs)
- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **Issues**: Check the app's error messages for specific issues

## Rollback

If you need to revert to CSV files:
1. Stop using Supabase (delete `.env` or set empty credentials)
2. The app will automatically fall back to CSV files
3. Your existing CSV files remain untouched

---

**🎉 Congratulations!** Your hog farm data is now in the cloud with professional-grade reliability and performance.
