/**
 * TATVA Marketing Website Client Script
 * Handles contact form submissions with validation, status feedback, and honeypot protection.
 */

document.addEventListener('DOMContentLoaded', () => {
  const contactForm = document.getElementById('contactForm');
  const submitBtn = document.getElementById('submitBtn');
  const formAlert = document.getElementById('formAlert');

  const nameInput = document.getElementById('name');
  const emailInput = document.getElementById('email');
  const subjectInput = document.getElementById('subject');
  const messageInput = document.getElementById('message');
  const honeypotInput = document.getElementById('website_url_hp');

  const nameError = document.getElementById('nameError');
  const emailError = document.getElementById('emailError');
  const subjectError = document.getElementById('subjectError');
  const messageError = document.getElementById('messageError');

  if (!contactForm) return;

  contactForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Reset errors & alerts
    clearErrors();
    showAlert('', '');

    // 1. Client-side Validation
    let isValid = true;

    if (!nameInput.value.trim()) {
      showFieldError(nameError, 'Name is required.');
      isValid = false;
    } else if (nameInput.value.trim().length > 100) {
      showFieldError(nameError, 'Name must not exceed 100 characters.');
      isValid = false;
    }

    if (!emailInput.value.trim()) {
      showFieldError(emailError, 'Email address is required.');
      isValid = false;
    } else if (!isValidEmail(emailInput.value.trim())) {
      showFieldError(emailError, 'Please enter a valid email address.');
      isValid = false;
    } else if (emailInput.value.trim().length > 255) {
      showFieldError(emailError, 'Email must not exceed 255 characters.');
      isValid = false;
    }

    if (!subjectInput.value.trim()) {
      showFieldError(subjectError, 'Subject is required.');
      isValid = false;
    } else if (subjectInput.value.trim().length > 200) {
      showFieldError(subjectError, 'Subject must not exceed 200 characters.');
      isValid = false;
    }

    if (!messageInput.value.trim()) {
      showFieldError(messageError, 'Message is required.');
      isValid = false;
    } else if (messageInput.value.trim().length > 5000) {
      showFieldError(messageError, 'Message must not exceed 5000 characters.');
      isValid = false;
    }

    if (!isValid) return;

    // 2. Prepare Payload (including honeypot)
    const payload = {
      name: nameInput.value.trim(),
      email: emailInput.value.trim(),
      subject: subjectInput.value.trim(),
      message: messageInput.value.trim(),
      website_url_hp: honeypotInput ? honeypotInput.value : ''
    };

    // 3. Set UI to Loading state
    setSubmitting(true);

    try {
      const response = await fetch('/api/contact', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (response.ok && data.success) {
        showAlert('success', data.message || 'Thank you! Your message has been sent successfully.');
        contactForm.reset();
      } else {
        showAlert('error', data.error || 'Failed to send message. Please try again.');
      }
    } catch (err) {
      console.error('Contact submit error:', err);
      showAlert('error', 'Network error. Please check your connection and try again.');
    } finally {
      setSubmitting(false);
    }
  });

  function isValidEmail(email) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  }

  function showFieldError(element, text) {
    if (element) {
      element.textContent = text;
    }
  }

  function clearErrors() {
    [nameError, emailError, subjectError, messageError].forEach(el => {
      if (el) el.textContent = '';
    });
  }

  function showAlert(type, message) {
    if (!formAlert) return;
    if (!message) {
      formAlert.className = 'alert hidden';
      formAlert.textContent = '';
      return;
    }
    formAlert.className = `alert alert-${type}`;
    formAlert.textContent = message;
  }

  function setSubmitting(isSubmitting) {
    if (!submitBtn) return;
    if (isSubmitting) {
      submitBtn.disabled = true;
      submitBtn.querySelector('.btn-text').textContent = 'Sending...';
    } else {
      submitBtn.disabled = false;
      submitBtn.querySelector('.btn-text').textContent = 'Send Message';
    }
  }
});
