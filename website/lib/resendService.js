/**
 * Resend email dispatch service for notifications and auto-replies.
 */

const { Resend } = require('resend');
const { escapeHtml } = require('./security');

class ResendService {
  constructor(apiKey, fromEmail, notificationEmail, mockClient = null) {
    this.apiKey = apiKey || process.env.RESEND_API_KEY;
    this.fromEmail = fromEmail || process.env.RESEND_FROM_EMAIL || 'TATVA <onboarding@resend.dev>';
    this.notificationEmail = notificationEmail || process.env.NOTIFICATION_EMAIL || 'team@tatvacompiler.com';

    if (mockClient) {
      this.client = mockClient;
    } else if (this.apiKey) {
      this.client = new Resend(this.apiKey);
    } else {
      this.client = null;
    }
  }

  /**
   * Sends notification to internal team and auto-reply to visitor.
   * ALL user text is HTML-escaped before interpolation into HTML email body.
   * 
   * @param {object} payload - Unescaped user submission ({ name, email, subject, message })
   * @returns {Promise<{ success: boolean, notificationResult?: object, autoReplyResult?: object, error?: string }>}
   */
  async sendContactEmails({ name, email, subject, message }) {
    // SECURITY CRITICAL: HTML escape user input before template interpolation
    const safeName = escapeHtml(name);
    const safeEmail = escapeHtml(email);
    const safeSubject = escapeHtml(subject);
    const safeMessage = escapeHtml(message);

    const notificationHtml = `
      <div style="font-family: Arial, sans-serif; color: #111; max-width: 600px; margin: 0 auto; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden;">
        <div style="background-color: #0B0F19; color: #10B981; padding: 20px; text-align: center;">
          <h2 style="margin: 0;">TATVA Compiler Contact Notification</h2>
        </div>
        <div style="padding: 24px;">
          <p><strong>From:</strong> ${safeName} (&lt;${safeEmail}&gt;)</p>
          <p><strong>Subject:</strong> ${safeSubject}</p>
          <hr style="border: 0; border-top: 1px solid #eee; margin: 16px 0;" />
          <p><strong>Message Payload:</strong></p>
          <div style="white-space: pre-wrap; background: #f8fafc; padding: 16px; border-left: 4px solid #10B981; border-radius: 4px; font-family: monospace;">${safeMessage}</div>
        </div>
      </div>
    `;

    const autoReplyHtml = `
      <div style="font-family: Arial, sans-serif; color: #111; max-width: 600px; margin: 0 auto; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden;">
        <div style="background-color: #0B0F19; color: #10B981; padding: 20px; text-align: center;">
          <h2 style="margin: 0;">TATVA — Message Received</h2>
        </div>
        <div style="padding: 24px;">
          <p>Hi ${safeName},</p>
          <p>Thank you for reaching out to the TATVA engineering team regarding <strong>${safeSubject}</strong>.</p>
          <p>We have received your message and will review it promptly.</p>
          <hr style="border: 0; border-top: 1px solid #eee; margin: 16px 0;" />
          <p style="font-size: 13px; color: #666;">A copy of your submission reference:</p>
          <blockquote style="margin: 0; padding: 12px; background-color: #f8fafc; border-left: 3px solid #06B6D4; font-size: 13px; white-space: pre-wrap;">${safeMessage}</blockquote>
          <p style="margin-top: 24px;">Best regards,<br><strong>TATVA Compiler Team</strong></p>
        </div>
      </div>
    `;

    if (!this.client) {
      console.warn('[ResendService] No API key provided. Operating in dry-run mode.');
      return {
        success: true,
        notificationResult: { id: 'mock-notif-' + Date.now(), html: notificationHtml },
        autoReplyResult: { id: 'mock-autoreply-' + Date.now(), html: autoReplyHtml }
      };
    }

    try {
      // 1. Company Notification Email
      const notificationPromise = this.client.emails.send({
        from: this.fromEmail,
        to: [this.notificationEmail],
        subject: `[Contact Form] ${subject}`,
        html: notificationHtml
      });

      // 2. Visitor Auto-Reply Email
      const autoReplyPromise = this.client.emails.send({
        from: this.fromEmail,
        to: [email],
        subject: `We received your message: ${subject}`,
        html: autoReplyHtml
      });

      const [notificationRes, autoReplyRes] = await Promise.all([
        notificationPromise,
        autoReplyPromise
      ]);

      return {
        success: true,
        notificationResult: notificationRes,
        autoReplyResult: autoReplyRes,
        // Return interpolated HTML templates so tests can assert escaping!
        renderedNotificationHtml: notificationHtml,
        renderedAutoReplyHtml: autoReplyHtml
      };
    } catch (err) {
      console.error('[ResendService] Failed to send emails:', err.message);
      return {
        success: false,
        error: err.message
      };
    }
  }
}

module.exports = ResendService;
