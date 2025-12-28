const express = require('express');
const { register, verifyEmail, resendVerification, login } = require('../controllers/authController');

const router = express.Router();

router.post('/register', register);
router.post('/verify-email', verifyEmail);
router.post('/resend-code', resendVerification);
router.post('/login', login);

module.exports = router;
