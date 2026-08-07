/**
 * Main contact form request handler coordinating validation, rate limiting,
 * database storage in Supabase, and transactional email delivery via Resend.
 */

const { validateContactInput } = require('./security');
const EmailRateLimiter = require('./rateLimiter');
const SupabaseService = require('./supabaseService');
const ResendService = require('./resendService');

class ContactHandler {
  constructor(options = {}) {
    this.rateLimiter = options.rateLimiter || new EmailRateLimiter(options.maxRequests || 3, options.windowMs || 15 * 60 * 1000);
    this.supabaseService = options.supabaseService || new SupabaseService(options.supabaseUrl, options.supabaseKey, options.mockSupabaseClient);
    this.resendService = options.resendService || new ResendService(options.resendApiKey, options.resendFromEmail, options.notificationEmail, options.mockResendClient);
  }

  /**
   * Handle incoming POST payload.
   * 
   * @param {object} payload - Request body
   * @param {object} metadata - Extra details (ipAddress, userAgent)
   * @returns {Promise<{ statusCode: number, body: object }>}
   */
  async handle(payload, metadata = {}) {
    // 1. Validate Input & Check Honeypot
    const validation = validateContactInput(payload);
    if (!validation.valid) {
      if (validation.isSpam) {
        // Honeypot triggered
        return {
          statusCode: 400,
          body: { success: false, error: 'Spam submission rejected.' }
        };
      }
      return {
        statusCode: 400,
        body: { success: false, error: validation.error }
      };
    }

    const { name, email, subject, message } = payload;

    // 2. Per-Email Rate Limiting Check
    const isLimited = this.rateLimiter.checkAndRecord(email);
    if (isLimited) {
      return {
        statusCode: 429,
        body: { success: false, error: 'Too many requests from this email address. Please try again later.' }
      };
    }

    // 3. Store Submission in Supabase
    const dbResult = await this.supabaseService.storeSubmission({
      name,
      email,
      subject,
      message,
      ipAddress: metadata.ipAddress,
      userAgent: metadata.userAgent
    });

    if (!dbResult.success) {
      return {
        statusCode: 500,
        body: { success: false, error: 'Failed to record submission.' }
      };
    }

    // 4. Send Notification & Visitor Auto-Reply via Resend (HTML-escaped internally)
    const emailResult = await this.resendService.sendContactEmails({
      name,
      email,
      subject,
      message
    });

    if (!emailResult.success) {
      return {
        statusCode: 500,
        body: { success: false, error: 'Submission recorded, but email notification failed.' }
      };
    }

    return {
      statusCode: 200,
      body: {
        success: true,
        message: 'Your message has been sent successfully! Check your inbox for confirmation.',
        dbRecord: dbResult.data,
        renderedNotificationHtml: emailResult.renderedNotificationHtml,
        renderedAutoReplyHtml: emailResult.renderedAutoReplyHtml
      }
    };
  }
}

module.exports = ContactHandler;
