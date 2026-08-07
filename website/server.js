/**
 * TATVA Marketing Website & Contact API Server
 */

const express = require('express');
const path = require('path');
const dotenv = require('dotenv');

// Load environment variables from .env file
dotenv.config({ path: path.join(__dirname, '.env') });
dotenv.config({ path: path.join(__dirname, '..', '.env') });

const ContactHandler = require('./lib/contactHandler');

const app = express();
const PORT = process.env.PORT || 3000;

// Body parser middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static frontend assets
app.use(express.static(__dirname));

// Initialize contact form handler
const contactHandler = new ContactHandler({
  supabaseUrl: process.env.SUPABASE_URL,
  supabaseKey: process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_KEY,
  resendApiKey: process.env.RESEND_API_KEY,
  resendFromEmail: process.env.RESEND_FROM_EMAIL,
  notificationEmail: process.env.NOTIFICATION_EMAIL
});

// Contact API Endpoint
app.post('/api/contact', async (req, res) => {
  try {
    const metadata = {
      ipAddress: req.ip || req.headers['x-forwarded-for'] || req.socket.remoteAddress,
      userAgent: req.headers['user-agent'] || 'Unknown'
    };

    const result = await contactHandler.handle(req.body, metadata);
    res.status(result.statusCode).json(result.body);
  } catch (err) {
    console.error('Unhandled server error in /api/contact:', err);
    res.status(500).json({ success: false, error: 'Internal server error.' });
  }
});

// Serve main page for any unknown routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Export app for testing and start server if executed directly
if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`[TATVA Marketing] Server listening on http://localhost:${PORT}`);
  });
}

module.exports = app;
