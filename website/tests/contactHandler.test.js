/**
 * Backend Unit & Integration Tests for TATVA Contact Form Handler
 * Tests: Validation, Honeypot, Rate Limiting, Supabase Storage, Resend Email Dispatch, and HTML Escaping.
 */

const { describe, it, beforeEach } = require('node:test');
const assert = require('node:assert/strict');

const ContactHandler = require('../lib/contactHandler');
const { escapeHtml, validateContactInput } = require('../lib/security');
const EmailRateLimiter = require('../lib/rateLimiter');

describe('TATVA Contact Form Backend Tests', () => {
  let mockSupabaseStore;
  let mockResendSend;
  let handler;

  beforeEach(() => {
    mockSupabaseStore = [];
    mockResendSend = [];

    // Mock Supabase Client
    const mockSupabaseClient = {
      from: (table) => ({
        insert: (rows) => ({
          select: async () => {
            mockSupabaseStore.push(...rows);
            return { data: rows.map((r, i) => ({ id: `row-${i + 1}`, ...r })), error: null };
          }
        })
      })
    };

    // Mock Resend Client
    const mockResendClient = {
      emails: {
        send: async (options) => {
          mockResendSend.push(options);
          return { id: `email-${Date.now()}-${Math.random()}` };
        }
      }
    };

    // Instantiate ContactHandler with mocks & fresh rate limiter
    handler = new ContactHandler({
      mockSupabaseClient,
      mockResendClient,
      maxRequests: 3,
      windowMs: 60 * 1000 // 1 min window for tests
    });
  });

  it('1. Valid submission stores in Supabase & dispatches emails', async () => {
    const payload = {
      name: 'Alice Turing',
      email: 'alice@riscv-research.org',
      subject: 'Compiler Schedules',
      message: 'Interested in Schraudolph fast softmax kernel benchmark numbers.'
    };

    const res = await handler.handle(payload, { ipAddress: '192.168.1.1', userAgent: 'TestRunner' });

    assert.equal(res.statusCode, 200);
    assert.equal(res.body.success, true);

    // Verify Supabase store received payload
    assert.equal(mockSupabaseStore.length, 1);
    assert.equal(mockSupabaseStore[0].name, 'Alice Turing');
    assert.equal(mockSupabaseStore[0].email, 'alice@riscv-research.org');
    assert.equal(mockSupabaseStore[0].ip_address, '192.168.1.1');

    // Verify Resend dispatched 2 emails (Company Notification + Visitor Auto-Reply)
    assert.equal(mockResendSend.length, 2);
    const recipients = mockResendSend.map(e => e.to[0]);
    assert.ok(recipients.includes('team@tatvacompiler.com'));
    assert.ok(recipients.includes('alice@riscv-research.org'));
  });

  it('2. Invalid inputs fail validation with proper errors', async () => {
    // Missing required fields
    const resMissing = await handler.handle({});
    assert.equal(resMissing.statusCode, 400);
    assert.equal(resMissing.body.success, false);
    assert.ok(resMissing.body.error.includes('required'));

    // Malformed email format
    const resBadEmail = await handler.handle({
      name: 'Bob',
      email: 'not-an-email',
      subject: 'Subject',
      message: 'Hello'
    });
    assert.equal(resBadEmail.statusCode, 400);
    assert.equal(resBadEmail.body.error, 'Invalid email address format.');

    // Message too long (> 5000 chars)
    const longMsg = 'A'.repeat(5001);
    const resLongMsg = await handler.handle({
      name: 'Bob',
      email: 'bob@example.com',
      subject: 'Subject',
      message: longMsg
    });
    assert.equal(resLongMsg.statusCode, 400);
    assert.ok(resLongMsg.body.error.includes('5000 characters'));
  });

  it('3. Honeypot filled submission is rejected instantly', async () => {
    const payload = {
      name: 'Spam Bot',
      email: 'spammer@botnet.com',
      subject: 'Buy Cheap Hardware',
      message: 'Click this link now!',
      website_url_hp: 'https://spam-link.com' // Honeypot trap filled
    };

    const res = await handler.handle(payload);

    assert.equal(res.statusCode, 400);
    assert.equal(res.body.success, false);
    assert.ok(res.body.error.includes('Spam'));

    // Ensure NO records stored in DB and NO emails sent
    assert.equal(mockSupabaseStore.length, 0);
    assert.equal(mockResendSend.length, 0);
  });

  it('4. Rate limit triggers on repeat submissions from same email', async () => {
    const email = 'repeater@example.com';
    const payload = {
      name: 'Repeater',
      email: email,
      subject: 'Inquiry',
      message: 'Repeated attempt'
    };

    // Submissions 1, 2, 3 should succeed
    for (let i = 1; i <= 3; i++) {
      const res = await handler.handle(payload);
      assert.equal(res.statusCode, 200, `Submission ${i} failed`);
    }

    // Submission 4 should trigger 429 Rate Limit
    const resRateLimited = await handler.handle(payload);
    assert.equal(resRateLimited.statusCode, 429);
    assert.equal(resRateLimited.body.success, false);
    assert.ok(resRateLimited.body.error.includes('Too many requests'));
  });

  it('5. Unescaped HTML input is HTML-escaped before email template interpolation', async () => {
    const maliciousPayload = {
      name: '<script>alert("xss")</script>',
      email: 'hacker@malicious.com',
      subject: 'Test & Query <foo>',
      message: '<img src="x" onerror="alert(1)"> & "quotes" \'single\''
    };

    const res = await handler.handle(maliciousPayload);
    assert.equal(res.statusCode, 200);

    // Verify rendered notification HTML has no raw script or unescaped tags
    const notifHtml = res.body.renderedNotificationHtml;
    const autoReplyHtml = res.body.renderedAutoReplyHtml;

    assert.ok(!notifHtml.includes('<script>alert("xss")</script>'));
    assert.ok(notifHtml.includes('&lt;script&gt;alert(&quot;xss&quot;)&lt;&#x2F;script&gt;'));
    assert.ok(notifHtml.includes('Test &amp; Query &lt;foo&gt;'));
    assert.ok(notifHtml.includes('&lt;img src=&quot;x&quot; onerror=&quot;alert(1)&quot;&gt; &amp; &quot;quotes&quot; &#39;single&#39;'));

    assert.ok(!autoReplyHtml.includes('<script>alert("xss")</script>'));
    assert.ok(autoReplyHtml.includes('&lt;script&gt;alert(&quot;xss&quot;)&lt;&#x2F;script&gt;'));
  });

  it('6. HTML escape utility function works standalone', () => {
    assert.equal(escapeHtml('<script>'), '&lt;script&gt;');
    assert.equal(escapeHtml('A & B'), 'A &amp; B');
    assert.equal(escapeHtml('"hello"'), '&quot;hello&quot;');
    assert.equal(escapeHtml("'world'"), '&#39;world&#39;');
    assert.equal(escapeHtml('a/b'), 'a&#x2F;b');
  });
});
