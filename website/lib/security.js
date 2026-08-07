/**
 * Security and validation utilities for contact form processing.
 */

/**
 * HTML-escapes all user-submitted text BEFORE interpolating into HTML email bodies.
 * Prevents HTML and script injection in email clients.
 * 
 * @param {string} str - Raw input text
 * @returns {string} - HTML-escaped string
 */
function escapeHtml(str) {
  if (typeof str !== 'string') {
    return '';
  }
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
    .replace(/\//g, '&#x2F;');
}

/**
 * Validates contact form submission payload.
 * 
 * @param {object} body - Form payload containing name, email, subject, message, website_url_hp
 * @returns {{ valid: boolean, error?: string, isSpam?: boolean }}
 */
function validateContactInput(body) {
  if (!body || typeof body !== 'object') {
    return { valid: false, error: 'Invalid payload.' };
  }

  // 1. Honeypot check
  if (body.website_url_hp && typeof body.website_url_hp === 'string' && body.website_url_hp.trim() !== '') {
    return { valid: false, isSpam: true, error: 'Spam submission detected.' };
  }

  // 2. Required field presence & type check
  const { name, email, subject, message } = body;

  if (!name || typeof name !== 'string' || name.trim() === '') {
    return { valid: false, error: 'Name is required.' };
  }
  if (!email || typeof email !== 'string' || email.trim() === '') {
    return { valid: false, error: 'Email is required.' };
  }
  if (!subject || typeof subject !== 'string' || subject.trim() === '') {
    return { valid: false, error: 'Subject is required.' };
  }
  if (!message || typeof message !== 'string' || message.trim() === '') {
    return { valid: false, error: 'Message is required.' };
  }

  // 3. Length checks
  const trimmedName = name.trim();
  const trimmedEmail = email.trim();
  const trimmedSubject = subject.trim();
  const trimmedMessage = message.trim();

  if (trimmedName.length > 100) {
    return { valid: false, error: 'Name must not exceed 100 characters.' };
  }
  if (trimmedEmail.length > 255) {
    return { valid: false, error: 'Email must not exceed 255 characters.' };
  }
  if (trimmedSubject.length > 200) {
    return { valid: false, error: 'Subject must not exceed 200 characters.' };
  }
  if (trimmedMessage.length > 5000) {
    return { valid: false, error: 'Message must not exceed 5000 characters.' };
  }

  // 4. Email format check
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(trimmedEmail)) {
    return { valid: false, error: 'Invalid email address format.' };
  }

  return { valid: true };
}

module.exports = {
  escapeHtml,
  validateContactInput
};
