const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const User = require('../models/User');
const { generateVerificationCode } = require('../utils/generateVerificationCode');
const path = require('path');
const Mailer = require(path.join(__dirname, '..', '..', '..', 'Mailer'));

const VERIFICATION_CODE_EXPIRY_MINUTES = 15;
const RESEND_COOLDOWN_SECONDS = 60;

function normalizeEmail(email) {
  return email.trim().toLowerCase();
}

function buildToken(userId) {
  const secret = process.env.JWT_SECRET;
  if (!secret) {
    throw new Error('JWT_SECRET environment variable is not set.');
  }

  return jwt.sign({ sub: userId }, secret, { expiresIn: '7d' });
}

async function sendVerificationEmail(email, code, name) {
  const subject = 'AiTrainer Hesap Doğrulama Kodunuz';
  const text = [
    `Merhaba ${name || 'Sporcu'},`,
    '',
    'AiTrainer hesabınızı doğrulamak için aşağıdaki kodu kullanın:',
    `Doğrulama Kodu: ${code}`,
    '',
    'Bu kod 15 dakika boyunca geçerlidir.',
    '',
    'Eğer bu isteği siz göndermediyseniz lütfen bu e-postayı göz ardı edin.',
    '',
    'AiTrainer Ekibi',
  ].join('\n');

  const html = `
    <div style="font-family: Arial, sans-serif; line-height: 1.5;">
      <p>Merhaba ${name || 'Sporcu'},</p>
      <p>AiTrainer hesabınızı doğrulamak için aşağıdaki kodu kullanın:</p>
      <p style="font-size: 20px; font-weight: bold; letter-spacing: 2px;">${code}</p>
      <p>Bu kod 15 dakika boyunca geçerlidir.</p>
      <p>Eğer bu isteği siz göndermediyseniz lütfen bu e-postayı göz ardı edin.</p>
      <p>AiTrainer Ekibi</p>
    </div>
  `;

  const mailSent = await Mailer.sendMail(email, subject, text, html);
  if (!mailSent) {
    throw new Error('Verification email could not be sent');
  }
}

async function register(req, res) {
  try {
    const { name, email, password } = req.body || {};

    if (!name || typeof name !== 'string' || name.trim().length < 2) {
      return res.status(400).json({ message: 'Geçerli bir isim girin.' });
    }

    if (!email || typeof email !== 'string' || !/\S+@\S+\.\S+/.test(email)) {
      return res.status(400).json({ message: 'Geçerli bir e-posta adresi girin.' });
    }

    if (!password || typeof password !== 'string' || password.length < 6) {
      return res.status(400).json({ message: 'Şifreniz en az 6 karakter olmalıdır.' });
    }

    const normalizedEmail = normalizeEmail(email);
    const existingUser = await User.findOne({ email: normalizedEmail });

    if (existingUser && existingUser.isVerified) {
      return res.status(409).json({ message: 'Bu e-posta adresi zaten kayıtlı.' });
    }

    if (
      existingUser &&
      existingUser.lastVerificationEmailSentAt &&
      (Date.now() - existingUser.lastVerificationEmailSentAt.getTime()) / 1000 < RESEND_COOLDOWN_SECONDS
    ) {
      const secondsLeft =
        RESEND_COOLDOWN_SECONDS -
        Math.floor((Date.now() - existingUser.lastVerificationEmailSentAt.getTime()) / 1000);
      return res
        .status(429)
        .json({ message: `Doğrulama kodu zaten gönderildi. ${secondsLeft} saniye sonra tekrar deneyin.` });
    }

    const passwordHash = await bcrypt.hash(password, 12);
    const verificationCode = generateVerificationCode();
    const verificationCodeHash = await bcrypt.hash(verificationCode, 10);
    const verificationCodeExpiresAt = new Date(Date.now() + VERIFICATION_CODE_EXPIRY_MINUTES * 60 * 1000);

    let user;
    if (existingUser) {
      existingUser.name = name.trim();
      existingUser.passwordHash = passwordHash;
      existingUser.isVerified = false;
      existingUser.verificationCodeHash = verificationCodeHash;
      existingUser.verificationCodeExpiresAt = verificationCodeExpiresAt;
      existingUser.lastVerificationEmailSentAt = new Date();
      user = await existingUser.save();
    } else {
      user = await User.create({
        name: name.trim(),
        email: normalizedEmail,
        passwordHash,
        isVerified: false,
        verificationCodeHash,
        verificationCodeExpiresAt,
        lastVerificationEmailSentAt: new Date(),
      });
    }

    await sendVerificationEmail(user.email, verificationCode, user.name);

    return res.status(201).json({
      message: 'Doğrulama kodu e-posta adresinize gönderildi.',
    });
  } catch (error) {
    console.error('register error:', error.message);
    return res.status(500).json({ message: 'Kayıt işlemi sırasında bir hata oluştu.' });
  }
}

async function verifyEmail(req, res) {
  try {
    const { email, code } = req.body || {};

    if (!email || typeof email !== 'string') {
      return res.status(400).json({ message: 'E-posta adresi gerekli.' });
    }

    if (!code || typeof code !== 'string') {
      return res.status(400).json({ message: 'Doğrulama kodu gerekli.' });
    }

    const user = await User.findOne({ email: normalizeEmail(email) });

    if (!user) {
      return res.status(404).json({ message: 'Kullanıcı bulunamadı.' });
    }

    if (user.isVerified) {
      return res.status(200).json({ message: 'Hesabınız zaten doğrulandı.' });
    }

    if (!user.verificationCodeHash || !user.verificationCodeExpiresAt) {
      return res.status(400).json({ message: 'Aktif bir doğrulama kodu bulunamadı.' });
    }

    if (user.verificationCodeExpiresAt.getTime() < Date.now()) {
      return res.status(400).json({ message: 'Doğrulama kodunun süresi dolmuş.' });
    }

    const isCodeValid = await bcrypt.compare(code, user.verificationCodeHash);
    if (!isCodeValid) {
      return res.status(400).json({ message: 'Geçersiz doğrulama kodu.' });
    }

    user.isVerified = true;
    user.verificationCodeHash = undefined;
    user.verificationCodeExpiresAt = undefined;
    user.lastVerificationEmailSentAt = undefined;
    await user.save();

    return res.status(200).json({ message: 'Hesabınız doğrulandı.' });
  } catch (error) {
    console.error('verifyEmail error:', error.message);
    return res.status(500).json({ message: 'Doğrulama işlemi sırasında bir hata oluştu.' });
  }
}

async function resendVerification(req, res) {
  try {
    const { email } = req.body || {};

    if (!email || typeof email !== 'string') {
      return res.status(400).json({ message: 'E-posta adresi gerekli.' });
    }

    const user = await User.findOne({ email: normalizeEmail(email) });

    if (!user) {
      return res.status(404).json({ message: 'Kullanıcı bulunamadı.' });
    }

    if (user.isVerified) {
      return res.status(400).json({ message: 'Hesabınız zaten doğrulandı.' });
    }

    if (
      user.lastVerificationEmailSentAt &&
      (Date.now() - user.lastVerificationEmailSentAt.getTime()) / 1000 < RESEND_COOLDOWN_SECONDS
    ) {
      const secondsLeft =
        RESEND_COOLDOWN_SECONDS -
        Math.floor((Date.now() - user.lastVerificationEmailSentAt.getTime()) / 1000);
      return res
        .status(429)
        .json({ message: `Doğrulama kodu zaten gönderildi. ${secondsLeft} saniye sonra tekrar deneyin.` });
    }

    const verificationCode = generateVerificationCode();
    user.verificationCodeHash = await bcrypt.hash(verificationCode, 10);
    user.verificationCodeExpiresAt = new Date(Date.now() + VERIFICATION_CODE_EXPIRY_MINUTES * 60 * 1000);
    user.lastVerificationEmailSentAt = new Date();
    await user.save();

    await sendVerificationEmail(user.email, verificationCode, user.name);

    return res.status(200).json({ message: 'Yeni doğrulama kodu gönderildi.' });
  } catch (error) {
    console.error('resendVerification error:', error.message);
    return res.status(500).json({ message: 'Doğrulama kodu gönderilirken bir hata oluştu.' });
  }
}

async function login(req, res) {
  try {
    const { email, password } = req.body || {};

    if (!email || typeof email !== 'string' || !password || typeof password !== 'string') {
      return res.status(400).json({ message: 'E-posta ve şifre gereklidir.' });
    }

    const user = await User.findOne({ email: normalizeEmail(email) });

    if (!user) {
      return res.status(401).json({ message: 'E-posta veya şifre hatalı.' });
    }

    if (!user.isVerified) {
      return res.status(403).json({ message: 'Lütfen önce e-posta adresinizi doğrulayın.' });
    }

    const isPasswordValid = await bcrypt.compare(password, user.passwordHash);

    if (!isPasswordValid) {
      return res.status(401).json({ message: 'E-posta veya şifre hatalı.' });
    }

    const token = buildToken(user._id.toString());

    return res.status(200).json({
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
      },
    });
  } catch (error) {
    console.error('login error:', error.message);
    return res.status(500).json({ message: 'Giriş işlemi sırasında bir hata oluştu.' });
  }
}

module.exports = {
  register,
  verifyEmail,
  resendVerification,
  login,
};
