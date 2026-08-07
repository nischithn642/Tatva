/**
 * In-memory rate limiter per submitting email address.
 */

class EmailRateLimiter {
  /**
   * @param {number} maxRequests - Max requests allowed within window (default: 3)
   * @param {number} windowMs - Time window in milliseconds (default: 15 mins = 900,000ms)
   */
  constructor(maxRequests = 3, windowMs = 15 * 60 * 1000) {
    this.maxRequests = maxRequests;
    this.windowMs = windowMs;
    this.records = new Map();
  }

  /**
   * Checks whether the given email address has exceeded the rate limit.
   * If limit is not reached, increments counter and returns false (not rate limited).
   * 
   * @param {string} email - Submitting email address
   * @returns {boolean} - True if rate limited, false otherwise
   */
  checkAndRecord(email) {
    if (!email || typeof email !== 'string') {
      return false;
    }

    const key = email.trim().toLowerCase();
    const now = Date.now();
    const userRecord = this.records.get(key) || { timestamps: [] };

    // Filter out timestamps outside the current sliding window
    userRecord.timestamps = userRecord.timestamps.filter(
      ts => now - ts < this.windowMs
    );

    if (userRecord.timestamps.length >= this.maxRequests) {
      return true; // Rate limited!
    }

    userRecord.timestamps.push(now);
    this.records.set(key, userRecord);
    return false;
  }

  /**
   * Resets rate limits (useful for testing).
   */
  reset() {
    this.records.clear();
  }
}

module.exports = EmailRateLimiter;
