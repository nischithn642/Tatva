# Supabase Database & Resend Integration Setup Guide

This document provides step-by-step instructions for configuring **Supabase** (Postgres storage with Row-Level Security) and **Resend** (transactional email service) for the TATVA contact form backend.

---

## 1. Supabase Database Setup

### Step 1: Create Table
In your Supabase project SQL Editor, execute the following SQL script to create the `contact_submissions` table:

```sql
-- Create contact_submissions table
CREATE TABLE IF NOT EXISTS public.contact_submissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL,
    subject VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    user_agent TEXT,
    ip_address VARCHAR(45),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for searching submissions by email and created date
CREATE INDEX IF NOT EXISTS idx_contact_submissions_email ON public.contact_submissions(email);
CREATE INDEX IF NOT EXISTS idx_contact_submissions_created_at ON public.contact_submissions(created_at DESC);
```

### Step 2: Enable Row-Level Security (RLS)
Enable Row-Level Security to prevent unauthorized public reading or deletion of submissions:

```sql
-- Enable RLS on contact_submissions table
ALTER TABLE public.contact_submissions ENABLE ROW LEVEL SECURITY;

-- Policy 1: Allow server-side service role full access (bypass RLS)
-- Note: Supabase service_role key automatically bypasses RLS policies.

-- Policy 2 (Optional for Anon key inserts): Allow anonymous submissions if using anon key
CREATE POLICY "Allow anonymous submission inserts" 
ON public.contact_submissions 
FOR INSERT 
TO anon 
WITH CHECK (true);

-- Deny public select/update/delete by default (No SELECT policy for anon)
```

---

## 2. Resend Transactional Email Setup

### Step 1: Add and Verify Domain
1. Log in to [Resend Dashboard](https://resend.com).
2. Go to **Domains** -> **Add Domain**.
3. Enter your domain (e.g., `tatvacompiler.com`).
4. Add the provided **MX**, **TXT (SPF & DKIM)**, and **CNAME (DMARC)** records to your DNS provider (e.g., Cloudflare, Route53).
5. Click **Verify Domain** until status becomes **Verified**.

### Step 2: Generate API Key
1. Go to **API Keys** -> **Create API Key**.
2. Name: `TATVA Production Website`.
3. Permission: `Sending access`.
4. Copy the generated key (`re_...`).

---

## 3. Environment Variables Configuration

Create a `.env` file in `website/.env` or root `.env`:

```ini
# Supabase Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.your_service_role_key

# Resend Configuration
RESEND_API_KEY=re_123456789_abcdefg
RESEND_FROM_EMAIL=TATVA Team <contact@tatvacompiler.com>
NOTIFICATION_EMAIL=team@tatvacompiler.com

# Server Configuration
PORT=3000
```

> [!CAUTION]
> NEVER commit `.env` or private keys (`SUPABASE_SERVICE_ROLE_KEY`, `RESEND_API_KEY`) to git repository.
