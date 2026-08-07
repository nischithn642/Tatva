/**
 * Supabase integration service for storing contact form submissions.
 */

const { createClient } = require('@supabase/supabase-js');

class SupabaseService {
  constructor(supabaseUrl, supabaseKey, mockClient = null) {
    this.supabaseUrl = supabaseUrl || process.env.SUPABASE_URL;
    this.supabaseKey = supabaseKey || process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_KEY;

    if (mockClient) {
      this.client = mockClient;
    } else if (this.supabaseUrl && this.supabaseKey) {
      this.client = createClient(this.supabaseUrl, this.supabaseKey);
    } else {
      this.client = null;
    }
  }

  /**
   * Insert contact submission into Supabase database.
   * 
   * @param {object} submission - Submission payload
   * @returns {Promise<{ success: boolean, data?: object, error?: string }>}
   */
  async storeSubmission(submission) {
    if (!this.client) {
      // In development/test without credentials, log and return mock success
      console.warn('[SupabaseService] No credentials provided. Operating in dry-run mode.');
      return {
        success: true,
        data: { id: 'mock-id-' + Date.now(), ...submission }
      };
    }

    try {
      const payload = {
        name: submission.name,
        email: submission.email,
        subject: submission.subject,
        message: submission.message,
        created_at: new Date().toISOString(),
        user_agent: submission.userAgent || 'Unknown',
        ip_address: submission.ipAddress || '127.0.0.1'
      };

      const { data, error } = await this.client
        .from('contact_submissions')
        .insert([payload])
        .select();

      if (error) {
        console.error('[SupabaseService] Insert error:', error.message);
        return { success: false, error: error.message };
      }

      return { success: true, data: data ? data[0] : payload };
    } catch (err) {
      console.error('[SupabaseService] Exception:', err.message);
      return { success: false, error: err.message };
    }
  }
}

module.exports = SupabaseService;
